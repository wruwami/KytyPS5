#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

void Check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename F>
void CheckFatal(F &&function, std::string_view expected, const char *message) {
  try {
    function();
  } catch (const std::runtime_error &error) {
    Check(std::string_view(error.what()).find(expected) !=
              std::string_view::npos,
          message);
    return;
  }
  Check(false, message);
}

struct Fixture {
  Program program;

  explicit Fixture(uint32_t block_count = 1) {
    program.stage = ShaderType::Compute;
    program.shader_hash = 0x12345678u;
    program.user_data_base = 2;
    for (uint32_t index = 0; index < block_count; index++) {
      program.block_storage.push_back(std::make_unique<Block>());
      auto *block = program.block_storage.back().get();
      program.blocks.push_back(block);
      program.block_info.push_back({.id = index});
    }
  }

  Block &BlockAt(uint32_t index = 0) { return *program.blocks[index]; }

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {},
             uint64_t flags = 0, uint32_t block = 0) {
    return Value(&BlockAt(block).AppendNewInst(opcode, args, flags));
  }

  Value EmitMemory(ValueOpcode opcode, std::initializer_list<Value> args,
                   uint32_t memory, uint32_t pc = 0x40, uint32_t block = 0) {
    MemoryFlags flags{.index = memory, .pc = pc};
    uint64_t bits = 0;
    std::memcpy(&bits, &flags, sizeof(flags));
    return Emit(opcode, args, bits, block);
  }

  uint32_t AddMemory(ResourceKind kind, int32_t offset = 0) {
    MemoryInfo info;
    info.kind = kind;
    info.offset = static_cast<uint32_t>(offset);
    program.memory_info.push_back(info);
    return static_cast<uint32_t>(program.memory_info.size() - 1u);
  }

  void Plan() { TrackResources(program, {}, {}); }
};

struct TestMemory {
  std::unordered_map<uint64_t, uint32_t> words;
  uint32_t reads = 0;
};

bool ReadMemory(void *userdata, uint64_t address, std::span<uint32_t> values) {
  auto &memory = *static_cast<TestMemory *>(userdata);
  for (auto &value : values) {
    const auto it = memory.words.find(address);
    if (it == memory.words.end()) return false;
    value = it->second;
    address += sizeof(uint32_t);
  }
  memory.reads++;
  return true;
}

Value Address(Fixture &fixture, Value low, Value high, uint32_t block = 0) {
  return fixture.Emit(ValueOpcode::GetAddressResource, {low, high}, 0, block);
}

Value RawRead(Fixture &fixture, Value address, Value offset, uint32_t memory,
              uint32_t block = 0) {
  return fixture.EmitMemory(ValueOpcode::LoadAddressU32,
                            {address, offset, Value(0u), Value(true)}, memory,
                            0x80, block);
}

void LoadBuffer(Fixture &fixture, std::array<Value, 4> words) {
  const auto handle = fixture.Emit(ValueOpcode::GetBufferResource,
                                   {words[0], words[1], words[2], words[3]});
  const auto loaded = fixture.EmitMemory(
      ValueOpcode::LoadBufferU32,
      {handle, Value(0u), Value(0u), Value(0u), Value(true)},
      fixture.AddMemory(ResourceKind::Buffer));
  fixture.Emit(ValueOpcode::ReferenceU32, {loaded});
}

void TestImmediateFlatteningAndGvn() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress, 0x20);
  const auto first = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  const auto second = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  LoadBuffer(fixture, {first, second, Value(16u), Value(0u)});

  fixture.Plan();
  Check(fixture.program.srt_reads.size() == 1,
        "equivalent typed scalar reads were not coalesced");
  Check(fixture.program.memory_info[memory].planning_only,
        "flattened raw read was not kept as a planning-only root");

  TestMemory memory_image{{{0x1020u, 0xfeedbeefu}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  Check(SrtWalker(fixture.program, runtime).RefreshFlatBuffer(flat), "flattened SRT walk failed");
  Check(flat == std::vector<uint32_t>{0xfeedbeefu} && memory_image.reads == 1,
        "flattened SRT did not evaluate its canonical read once");
}

void TestRawScalarComponentAlignment() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress, 2);
  const auto read = RawRead(
      fixture, Address(fixture, Value(0x1003u), Value(0u)), Value(2u), memory);
  LoadBuffer(fixture, {read, Value(0u), Value(16u), Value(0u)});
  fixture.Plan();

  TestMemory memory_image{{{0x1000u, 0x12345678u}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  Check(SrtWalker(fixture.program, runtime).RefreshFlatBuffer(flat), "raw scalar SRT walk failed");
  Check(
      flat == std::vector<uint32_t>{0x12345678u} && memory_image.reads == 1,
      "raw scalar base, immediate, and offset were not aligned independently");
}

void TestScalarMemoryDomainMismatchFails() {
  Fixture raw;
  const auto raw_memory = raw.AddMemory(ResourceKind::ScalarBuffer);
  RawRead(raw, Address(raw, Value(0x1000u), Value(0u)), Value(0u), raw_memory);
  CheckFatal([&] { TrackResources(raw.program, {}, {}); },
             "incompatible scalar memory metadata",
             "raw scalar load accepted descriptor-buffer metadata");

  Fixture buffer;
  const auto buffer_memory = buffer.AddMemory(ResourceKind::ScalarAddress);
  const auto resource =
      buffer.Emit(ValueOpcode::GetBufferResource,
                  {Value(0x1000u), Value(0u), Value(16u), Value(0u)});
  buffer.EmitMemory(ValueOpcode::ReadConstBuffer, {resource, Value(0u)},
                    buffer_memory);
  CheckFatal([&] { TrackResources(buffer.program, {}, {}); },
             "incompatible scalar memory metadata",
             "descriptor scalar load accepted raw-address metadata");
}

void TestDynamicReadRemainsTyped() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  const auto offset = fixture.Emit(ValueOpcode::GetUserData,
                                   {Value(static_cast<ScalarReg>(2))});
  const auto read = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), offset, memory);
  LoadBuffer(fixture, {read, Value(0u), Value(16u), Value(0u)});

  fixture.Plan();
  Check(fixture.program.srt_reads.empty() &&
            read.ResolveInstruction()->GetOpcode() == ValueOpcode::LoadAddressU32 &&
            !fixture.program.memory_info[memory].planning_only,
        "dynamic scalar read received a fake flattened slot");
}

void TestNestedSrtWalk() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  const auto pointer = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  const auto value =
      RawRead(fixture, Address(fixture, pointer, Value(0u)), Value(0u), memory);
  LoadBuffer(fixture, {value, Value(0u), Value(16u), Value(0u)});
  fixture.Plan();

  TestMemory memory_image{{{0x1000u, 0x2000u}, {0x2000u, 0xabcdef01u}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  Check(SrtWalker(fixture.program, runtime).RefreshFlatBuffer(flat), "nested SRT walk failed");
  Check(flat == std::vector<uint32_t>({0x2000u, 0xabcdef01u}),
        "nested typed SRT reads were not evaluated in dependency order");
}

void TestShaderBaseAndUserData() {
  Fixture fixture;
  const auto base = fixture.Emit(ValueOpcode::GetShaderBase);
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {base, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {base, Value(1u)});
  const auto user = fixture.Emit(ValueOpcode::GetUserData,
                                 {Value(static_cast<ScalarReg>(2))});
  const auto sum = fixture.Emit(ValueOpcode::IAdd32, {user, Value(4u)});

  fixture.program.descriptor_sources.push_back(
      {.dwords = {low, high, sum}, .dword_count = 3});

  const std::array user_data{0x20u};
  SrtRuntime runtime{.user_data = user_data,
                     .shader_base = 0x12345678abcdef00ull};
  DescriptorValue result;
  Check(SrtWalker(fixture.program, runtime).EvaluateDescriptor(0, result),
        "shader-relative descriptor evaluation failed");
  Check(result.dword_count == 3 && result.dwords[0] == 0xabcdef00u &&
            result.dwords[1] == 0x12345678u && result.dwords[2] == 0x24u,
        "shader-relative typed descriptor expression evaluated incorrectly");
}

void TestCarryAndBitFields() {
  Fixture fixture;
  const auto carry =
      fixture.Emit(ValueOpcode::IAddCarry32, {Value(0xffffffffu), Value(2u)});
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU32x2, {carry, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU32x2, {carry, Value(1u)});
  const auto inserted =
      fixture.Emit(ValueOpcode::BitFieldInsert,
                   {Value(0u), Value(0x89abcdefu), Value(0u), Value(32u)});
  const auto sign = fixture.Emit(ValueOpcode::BitFieldSExtract,
                                 {Value(0x000000f0u), Value(4u), Value(4u)});

  fixture.program.descriptor_sources.push_back(
      {.dwords = {low, high, inserted, sign}, .dword_count = 4});

  DescriptorValue result;
  Check(SrtWalker(fixture.program, {}).EvaluateDescriptor(0, result),
        "carry and bit-field descriptor evaluation failed");
  Check(result.dwords[0] == 1u && result.dwords[1] == 1u &&
            result.dwords[2] == 0x89abcdefu && result.dwords[3] == 0xffffffffu,
        "typed carry or bit-field runtime evaluation is incorrect");
}

void TestInvariantAndDivergentPhi() {
  Fixture fixture(3);
  auto &invariant = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  invariant.SetFlags(Type::U32);
  invariant.AddPhiOperand(&fixture.BlockAt(0), Value(7u));
  invariant.AddPhiOperand(&fixture.BlockAt(1), Value(7u));
  auto &divergent = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  divergent.SetFlags(Type::U32);
  divergent.AddPhiOperand(&fixture.BlockAt(0), Value(7u));
  divergent.AddPhiOperand(&fixture.BlockAt(1), Value(9u));

  fixture.program.descriptor_sources.push_back(
      {.dwords = {Value(&invariant)}, .dword_count = 1});
  fixture.program.descriptor_sources.push_back(
      {.dwords = {Value(&divergent)}, .dword_count = 1});

  DescriptorValue result;
  Check(SrtWalker(fixture.program, {}).EvaluateDescriptor(0, result) &&
            result.dwords[0] == 7u,
        "loop-invariant typed phi was rejected");
  Check(!SrtWalker(fixture.program, {}).EvaluateDescriptor(1, result),
        "divergent phi was accepted");
}

void TestOperandMutationLifecycle() {
  Fixture fixture(6);
  const auto definition = fixture.Emit(ValueOpcode::IAdd32, {Value(0x55u), Value(0u)});
  auto *source = definition.TryInstruction();
  const auto inserted = fixture.Emit(ValueOpcode::BitFieldInsert,
      {definition, definition, Value(4u), Value(4u)});
  Check(source->UseCount() == 2, "inline operands lost distinct reverse-use indices");
  inserted.TryInstruction()->SetArg(1, Value(3u));
  Check(source->UseCount() == 1, "inline operand mutation did not detach its old use");
  source->ReplaceUsesWith(Value(0xaau));
  uint32_t result = 0;
  Check(source->UseCount() == 0 &&
            SrtWalker(fixture.program, {}).Evaluate(inserted, result) && result == 0x3au,
        "inline operand replacement changed the evaluated value");

  for (const auto opcode : {ValueOpcode::GetImageResource, ValueOpcode::MakeImageAddress}) {
    auto &large = fixture.program.value_storage.emplace_back(opcode);
    auto &replacement = fixture.program.value_storage.emplace_back(opcode);
    const auto count = large.NumArgs();
    for (size_t index = 0; index < count; ++index) {
      large.SetArg(index, definition);
      replacement.SetArg(index, Value(static_cast<uint32_t>(index)));
    }
    const auto user = fixture.Emit(ValueOpcode::Identity, {Value(&large)}, 0, 1);
    Check(source->UseCount() == count, "large operands lost reverse-use indices");
    large.ReplaceUsesWith(Value(&replacement));
    Check(source->UseCount() == 0 && large.GetOpcode() == ValueOpcode::Identity &&
              large.NumArgs() == 1 && large.Arg(0) == Value(&replacement) &&
              user.Resolve() == Value(&replacement) && replacement.UseCount() == 2,
          "large operand replacement did not preserve users or detach old operands");
    large.Invalidate();
    large.Invalidate();
    Check(large.NumArgs() == 0 && replacement.UseCount() == 1,
          "repeated invalidation detached a surviving large-value user");
  }

  auto &phi = fixture.BlockAt(5).AppendNewInst(ValueOpcode::Phi);
  phi.SetFlags(Type::U32);
  for (size_t index = 0; index < 5; ++index) {
    phi.AddPhiOperand(&fixture.BlockAt(index), definition);
  }
  phi.SetArg(3, Value(7u));
  Check(phi.NumArgs() == 5 && phi.NumPhiBlocks() == 5 && source->UseCount() == 4,
        "Phi mutation lost its incoming values or reverse uses");
  for (size_t index = 0; index < 5; ++index) {
    Check(phi.PhiBlock(index) == &fixture.BlockAt(index),
          "Phi mutation changed an incoming predecessor");
  }
  const auto sum = fixture.Emit(ValueOpcode::IAdd32, {Value(&phi), Value(5u)}, 0, 5);
  phi.ReplaceUsesWith(Value(6u));
  Check(source->UseCount() == 0 && phi.NumPhiBlocks() == 0 && phi.NumArgs() == 1 &&
            SrtWalker(fixture.program, {}).Evaluate(sum, result) && result == 11,
        "Phi replacement retained incoming uses or changed the consumer value");
}

void TestControlDependentStandaloneLoadStaysTyped() {
  Fixture fixture(3);
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  auto &base = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  base.SetFlags(Type::U32);
  base.AddPhiOperand(&fixture.BlockAt(0), Value(0x1000u));
  base.AddPhiOperand(&fixture.BlockAt(1), Value(0x2000u));
  const auto read =
      RawRead(fixture, Address(fixture, Value(&base), Value(0u), 2), Value(0u),
              memory, 2);
  fixture.Emit(ValueOpcode::ReferenceU32, {read}, 0, 2);
  fixture.Plan();
  Check(fixture.program.srt_reads.empty() &&
            read.ResolveInstruction()->GetOpcode() ==
                ValueOpcode::LoadAddressU32 &&
            !fixture.program.memory_info[memory].planning_only,
        "control-dependent standalone scalar load was flattened into a host "
        "snapshot");
}

void TestRuntime64BitDescriptorOps() {
  Fixture fixture;
  const auto shifted = fixture.Emit(ValueOpcode::ShiftLeftLogical64,
                                    {Value(uint64_t{0x1234u}), Value(32u)});
  const auto masked =
      fixture.Emit(ValueOpcode::BitwiseAnd64,
                   {shifted, Value(uint64_t{0x0000ffff00000000ull})});
  const auto combined = fixture.Emit(
      ValueOpcode::IAdd64, {masked, Value(uint64_t{0x000000010000abcdull})});
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {combined, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {combined, Value(1u)});

  fixture.program.descriptor_sources.push_back(
      {.dwords = {low, high}, .dword_count = 2});

  DescriptorValue result;
  Check(SrtWalker(fixture.program, {}).EvaluateDescriptor(0, result) &&
            result.dwords[0] == 0xabcdu && result.dwords[1] == 0x1235u,
        "64-bit typed descriptor arithmetic evaluation is incorrect");
}

void TestUniformFirstLaneSamplerLod() {
  Fixture fixture;
  const auto active = fixture.Emit(
      ValueOpcode::IEqual32, {fixture.Emit(ValueOpcode::LaneId), Value(0u)});
  const auto stale = fixture.Emit(
      ValueOpcode::GetVectorRegister, {Value(static_cast<VectorReg>(0))});
  const auto write = [&](Value value, Value previous) {
    return fixture.Emit(ValueOpcode::SelectU32, {active, value, previous});
  };
  const auto user = fixture.Emit(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(8))});
  const auto as_float = fixture.Emit(ValueOpcode::ConvertF32U32, {user});
  const auto initial = write(
      fixture.Emit(ValueOpcode::BitCastU32F32, {as_float}), stale);
  const auto initial_float =
      fixture.Emit(ValueOpcode::BitCastF32U32, {initial});
  const auto scaled = fixture.Emit(ValueOpcode::FPMul32,
                                   {Value::F32(256.0f), initial_float});
  const auto scaled_write = write(
      fixture.Emit(ValueOpcode::BitCastU32F32, {scaled}), initial);
  const auto scaled_float =
      fixture.Emit(ValueOpcode::BitCastF32U32, {scaled_write});
  const auto nan = fixture.Emit(ValueOpcode::FPIsNan32, {scaled_float});
  const auto low = fixture.Emit(ValueOpcode::FPOrdLessThanEqual32,
                                {scaled_float, Value::F32(0.0f)});
  const auto high = fixture.Emit(ValueOpcode::FPOrdGreaterThanEqual32,
                                 {scaled_float, Value::F32(4294967296.0f)});
  const auto truncated = fixture.Emit(ValueOpcode::FPTrunc32, {scaled_float});
  const auto safe_low = fixture.Emit(
      ValueOpcode::SelectF32,
      {fixture.Emit(ValueOpcode::LogicalOr, {nan, low}), Value::F32(0.0f),
       truncated});
  const auto safe = fixture.Emit(
      ValueOpcode::SelectF32,
      {high, Value::F32(4294967040.0f), safe_low});
  const auto converted = fixture.Emit(ValueOpcode::ConvertU32F32, {safe});
  const auto saturated = write(
      fixture.Emit(ValueOpcode::SelectU32,
                   {high, Value(UINT32_MAX), converted}),
      scaled_write);
  const auto lod = fixture.Emit(ValueOpcode::UMin32,
                                {saturated, Value(0xfffu)});
  const auto lod_write = write(lod, saturated);
  const auto masked = fixture.Emit(ValueOpcode::BitwiseAnd32,
                                   {lod_write, Value(0xfffu)});
  const auto masked_write = write(masked, lod_write);
  const auto packed = write(
      fixture.Emit(
          ValueOpcode::BitwiseOr32,
          {fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                        {masked_write, Value(12u)}),
           masked_write}),
      masked_write);
  const auto first = fixture.Emit(ValueOpcode::ReadFirstLane,
                                  {packed, active});
  const auto divergent = fixture.Emit(
      ValueOpcode::ReadFirstLane, {stale, active});

  Check(ValidateRuntimeValue(fixture.program, first),
        "uniform sampler LOD construction was rejected");
  Check(!ValidateRuntimeValue(fixture.program, divergent),
        "divergent first-lane value was accepted as uniform");
  fixture.program.descriptor_sources.push_back(
      {.dwords = {first}, .dword_count = 1});

  std::array<uint32_t, 7> user_data{};
  user_data[6] = 3u;
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue result;
  Check(SrtWalker(fixture.program, runtime).EvaluateDescriptor(0, result) &&
            result.dwords[0] == 0x00300300u,
        "uniform sampler LOD evaluated incorrectly");
  user_data[6] = 20u;
  Check(SrtWalker(fixture.program, runtime).EvaluateDescriptor(0, result) &&
            result.dwords[0] == 0x00ffffffu,
        "uniform sampler LOD clamp evaluated incorrectly");
}

void TestFloatComparisonDescriptorInputs() {
  struct Input {
    uint32_t bits;
    uint32_t less_equal;
    uint32_t greater_equal;
  };
  constexpr std::array inputs = {
      Input{0x00000001u, 0, 1}, Input{0x80000001u, 1, 0},
      Input{0x007fffffu, 0, 1}, Input{0x807fffffu, 1, 0},
      Input{0x00800000u, 0, 1}, Input{0x80800000u, 1, 0},
      Input{0x00000000u, 1, 1}, Input{0x80000000u, 1, 1},
      Input{0x7fc00000u, 0, 0}};
  for (const bool flush : {false, true}) {
    Fixture fixture;
    const auto user = fixture.Emit(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(2))});
    const auto value = fixture.Emit(ValueOpcode::BitCastF32U32, {user});
    const auto less_equal = fixture.Emit(ValueOpcode::FPOrdLessThanEqual32,
                                         {value, Value::F32(0.0f)});
    const auto greater_equal = fixture.Emit(ValueOpcode::FPOrdGreaterThanEqual32,
                                            {value, Value::F32(0.0f)});
    less_equal.Instruction()->SetFlags(FPCompareFlags{flush});
    greater_equal.Instruction()->SetFlags(FPCompareFlags{flush});
    const auto low = fixture.Emit(ValueOpcode::SelectU32,
                                  {less_equal, Value(1u), Value(0u)});
    const auto high = fixture.Emit(ValueOpcode::SelectU32,
                                   {greater_equal, Value(1u), Value(0u)});
    fixture.program.descriptor_sources.push_back(
        {.dwords = {low, high}, .dword_count = 2});

    for (size_t index = 0; index < inputs.size(); index++) {
      const auto &input = inputs[index];
      const std::array user_data{input.bits};
      DescriptorValue result;
      Check(SrtWalker(fixture.program, {.user_data = user_data})
                    .EvaluateDescriptor(0, result) &&
                result.dwords[0] == (flush && index < 4 ? 1 : input.less_equal) &&
                result.dwords[1] == (flush && index < 4 ? 1 : input.greater_equal),
            "descriptor comparison disagrees with native FP32 input mode");
    }
  }
}

void TestSharedIntegerRuntimeDependencies() {
  Fixture fixture;
  const auto lane = fixture.Emit(ValueOpcode::LaneId);
  const auto active =
      fixture.Emit(ValueOpcode::IEqual32, {lane, Value(0u)});
  auto inactive = lane;
  for (uint32_t level = 0; level < 36; level++) {
    const auto left =
        fixture.Emit(ValueOpcode::IAdd32, {inactive, Value(1u)});
    const auto right =
        fixture.Emit(ValueOpcode::IMul32, {inactive, Value(3u)});
    inactive = fixture.Emit(ValueOpcode::BitwiseXor32, {left, right});
  }
  const auto selected = fixture.Emit(
      ValueOpcode::SelectU32, {active, Value(42u), inactive});
  const auto first =
      fixture.Emit(ValueOpcode::ReadFirstLane, {selected, active});
  Check(ValidateRuntimeValue(fixture.program, first, RuntimeValueType::Integer),
        "shared integer dependencies behind an inactive arm were rejected");

  const auto uniform_use = fixture.Emit(ValueOpcode::IAdd32, {first, inactive});
  Check(!ValidateRuntimeValue(fixture.program, uniform_use,
                              RuntimeValueType::Integer),
        "integer-only dependency acceptance was reused as uniform acceptance");

  const auto other_active =
      fixture.Emit(ValueOpcode::IEqual32, {lane, Value(1u)});
  const auto other_first =
      fixture.Emit(ValueOpcode::ReadFirstLane, {selected, other_active});
  const auto both = fixture.Emit(ValueOpcode::IAdd32, {first, other_first});
  Check(!ValidateRuntimeValue(fixture.program, both, RuntimeValueType::Integer),
        "uniform acceptance was reused across different execution masks");

  const auto floating =
      fixture.Emit(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  const auto mixed =
      fixture.Emit(ValueOpcode::BitwiseOr32, {inactive, floating});
  selected.ResolveInstruction()->SetArg(2, mixed);
  Check(!ValidateRuntimeValue(fixture.program, first, RuntimeValueType::Integer),
        "shared integer dependencies hid a floating-point sibling");
}

void TestConstantBufferBounds() {
  struct Case {
    uint32_t offset;
    uint32_t immediate;
    bool valid;
    uint32_t expected;
  };
  for (const auto &test : {Case{12u, 0u, true, 0xa5a5a5a5u},
                           Case{3u, 1u, true, 0x12345678u},
                           Case{0xfffffffcu, 4u, false, 0u},
                           Case{16u, 0u, false, 0u}}) {
    Fixture fixture;
    const auto memory = fixture.AddMemory(ResourceKind::ScalarBuffer, test.immediate);
    const auto buffer =
        fixture.Emit(ValueOpcode::GetBufferResource,
                     {Value(0x3000u), Value(0u), Value(16u), Value(0u)});
    const auto read = fixture.EmitMemory(ValueOpcode::ReadConstBuffer,
                                         {buffer, Value(test.offset)}, memory);
    LoadBuffer(fixture, {read, Value(0u), Value(16u), Value(0u)});
    fixture.Plan();

    TestMemory memory_image{{{0x3000u, 0x12345678u}, {0x300cu, 0xa5a5a5a5u}}};
    SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
    std::vector<uint32_t> flat;
    Check(SrtWalker(fixture.program, runtime).RefreshFlatBuffer(flat) == test.valid,
          "constant-buffer walk misaligned or wrapped its offset components");
    Check(test.valid ? flat == std::vector<uint32_t>{test.expected} : memory_image.reads == 0,
          "constant-buffer walk read the wrong word or accessed an out-of-bounds address");
  }
}

void TestReadLaneElimination() {
  Fixture fixture;
  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto write = fixture.Emit(ValueOpcode::WriteLane,
                                  {undef, Value(0xdeadbeefu), Value(5u)});
  const auto read = fixture.Emit(ValueOpcode::ReadLane, {write, Value(5u)});
  const auto use = fixture.Emit(ValueOpcode::IAdd32, {read, Value(1u)});
  const auto stats = EliminateReadLane(fixture.program, 64);
  Check(stats.rewritten_reads == 1 &&
            use.ResolveInstruction()->Arg(0).Resolve() == Value(0xdeadbeefu),
        "fixed-lane typed read was not rewritten from its SSA write chain");

  const auto selector = fixture.Emit(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(2))});
  const auto dynamic = fixture.Emit(ValueOpcode::ReadLane, {write, selector});
  fixture.Emit(ValueOpcode::IAdd32, {dynamic, Value(1u)});
  Check(EliminateReadLane(fixture.program, 64).rewritten_reads == 0,
        "dynamic-lane read was rewritten unsafely");
}

void TestOptimizationPipeline() {
  Fixture fixture;
  const auto sum = fixture.Emit(ValueOpcode::IAdd32, {Value(40u), Value(2u)});
  const auto kept = fixture.Emit(ValueOpcode::BitwiseOr32, {sum, Value(0u)});
  fixture.Emit(ValueOpcode::ReferenceU32, {kept});
  fixture.Emit(ValueOpcode::IMul32, {Value(6u), Value(7u)});

  ConstantPropagationPass(fixture.program.blocks);
  RemoveIdentities(fixture.program.blocks);
  EliminateDeadCode(fixture.program.blocks);

  const auto &instructions = fixture.BlockAt().Instructions();
  Check(instructions.size() == 1 &&
            instructions.front().GetOpcode() == ValueOpcode::ReferenceU32 &&
            instructions.front().Arg(0).Resolve() == Value(42u),
        "typed constant propagation, identity folding, or dead-code "
        "elimination regressed");
}

void TestControlFlowValueSurvivesReadLaneFolding() {
  Fixture fixture(3);
  auto *entry = fixture.program.blocks[0];
  auto *taken = fixture.program.blocks[1];
  auto *other = fixture.program.blocks[2];
  entry->AddBranch(taken);
  entry->AddBranch(other);

  auto &entry_info = fixture.program.block_info[0];
  entry_info.terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  entry_info.terminator.true_block = 1;
  entry_info.terminator.false_block = 2;
  fixture.program.block_info[1].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  fixture.program.block_info[2].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;

  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto write =
      fixture.Emit(ValueOpcode::WriteLane, {undef, Value(42u), Value(5u)});
  const auto read = fixture.Emit(ValueOpcode::ReadLane, {write, Value(5u)});
  entry_info.condition =
      fixture.Emit(ValueOpcode::IEqual32, {read, Value(42u)});
  fixture.Emit(ValueOpcode::Reference, {entry_info.condition});

  Check(EliminateReadLane(fixture.program, 64).rewritten_reads == 1,
        "control-flow fixture did not eliminate its fixed-lane read");
  ConstantPropagationPass(fixture.program.blocks);
  ResolveControlFlowIdentities(fixture.program);
  RemoveIdentities(fixture.program.blocks);
  EliminateDeadCode(fixture.program.blocks);

  Check(entry_info.condition == Value(true),
        "folded branch condition did not survive identity removal");
  ValidateProgram(fixture.program, true);
}

void TestDeadPhiCyclesAndPlanningRoots() {
  Fixture fixture(2);
  auto &entry = fixture.BlockAt(0);
  auto &loop = fixture.BlockAt(1);
  entry.AddBranch(&loop);
  loop.AddBranch(&loop);
  const auto source = fixture.Emit(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(2))});
  const auto identity = fixture.Emit(ValueOpcode::Identity, {source});
  auto &live = loop.AppendNewInst(ValueOpcode::Phi, {}, uint64_t(Type::U32));
  live.AddPhiOperand(&entry, identity);
  live.AddPhiOperand(&loop, Value(&live));
  const auto reference = fixture.Emit(ValueOpcode::ReferenceU32, {Value(&live)}, 0, 1);
  auto &dead = loop.AppendNewInst(ValueOpcode::Phi, {}, uint64_t(Type::U32));
  const auto increment = fixture.Emit(ValueOpcode::IAdd32, {Value(&dead), Value(1u)}, 0, 1);
  dead.AddPhiOperand(&entry, source);
  dead.AddPhiOperand(&loop, increment);
  const auto retained = fixture.Emit(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(3))});
  auto &planning = fixture.program.value_storage.emplace_back(ValueOpcode::Identity);
  planning.SetArg(0, retained);

  EliminateDeadCode(fixture.program.blocks);
  Check(entry.Instructions().size() == 3 && loop.Instructions().size() == 2,
        "dead Phi cycle survived or a live/planning dependency was removed");
  Check(identity.Instruction()->Arg(0) == source && live.Arg(0) == identity &&
            live.Arg(1) == Value(&live) && planning.Arg(0) == retained,
        "DCE did not preserve direct Identity operands or live Phi recurrence");
  // Dropping roots must clear prior marks and remove the newly dead SCC safely.
  reference.Instruction()->Invalidate();
  planning.Invalidate();
  EliminateDeadCode(fixture.program.blocks);
  Check(entry.empty() && loop.empty(), "DCE reused stale marks after removing its roots");
}

void TestUndefinedRuntimeValueFails() {
  Fixture fixture;
  const auto undef = fixture.Emit(ValueOpcode::UndefU32);

  fixture.program.descriptor_sources.push_back(
      {.dwords = {undef}, .dword_count = 1});
  DescriptorValue result;
  Check(!SrtWalker(fixture.program, {}).EvaluateDescriptor(0, result),
        "undefined typed descriptor source was accepted");
}

void TestConstantPropagationArithmeticAndBitwise() {
  Fixture fixture;
  // Identity cases need a non-immediate operand; otherwise the constant fold
  // fires first and the identity rule under test is never reached.
  const auto sym32 = fixture.Emit(ValueOpcode::UndefU32);
  const auto sym64 = fixture.Emit(ValueOpcode::UndefU64);
  // ISub32
  const auto sub32 = fixture.Emit(ValueOpcode::ISub32, {Value(50u), Value(8u)});
  const auto sub32_id = fixture.Emit(ValueOpcode::ISub32, {sym32, Value(0u)});
  // ISub64
  const auto sub64 = fixture.Emit(ValueOpcode::ISub64, {Value(uint64_t{100ull}), Value(uint64_t{42ull})});
  const auto sub64_id = fixture.Emit(ValueOpcode::ISub64, {sym64, Value(uint64_t{0ull})});
  // IMul64
  const auto mul64 = fixture.Emit(ValueOpcode::IMul64, {Value(uint64_t{6ull}), Value(uint64_t{7ull})});
  const auto mul64_id = fixture.Emit(ValueOpcode::IMul64, {sym64, Value(uint64_t{1ull})});
  // UDiv32
  const auto div32 = fixture.Emit(ValueOpcode::UDiv32, {Value(84u), Value(2u)});
  const auto div32_id = fixture.Emit(ValueOpcode::UDiv32, {Value(42u), Value(1u)});
  // SMulHi / UMulHi
  const auto smulhi = fixture.Emit(ValueOpcode::SMulHi, {Value(0x10000000u), Value(0x10000000u)});
  const auto umulhi = fixture.Emit(ValueOpcode::UMulHi, {Value(0x80000000u), Value(2u)});
  // IAbs32
  const auto abs_pos = fixture.Emit(ValueOpcode::IAbs32, {Value(42u)});
  const auto abs_neg = fixture.Emit(ValueOpcode::IAbs32, {Value(0x80000005u)});
  // Shifts
  const auto shl32 = fixture.Emit(ValueOpcode::ShiftLeftLogical32, {Value(1u), Value(4u)});
  const auto shl32_id = fixture.Emit(ValueOpcode::ShiftLeftLogical32, {sym32, Value(0u)});
  const auto shr32 = fixture.Emit(ValueOpcode::ShiftRightLogical32, {Value(32u), Value(2u)});
  const auto shr32_id = fixture.Emit(ValueOpcode::ShiftRightLogical32, {sym32, Value(0u)});
  const auto sra32 = fixture.Emit(ValueOpcode::ShiftRightArithmetic32, {Value(static_cast<uint32_t>(-16)), Value(2u)});
  const auto shl64 = fixture.Emit(ValueOpcode::ShiftLeftLogical64, {Value(uint64_t{1ull}), Value(40u)});
  const auto shl64_id = fixture.Emit(ValueOpcode::ShiftLeftLogical64, {sym64, Value(0u)});
  const auto shr64 = fixture.Emit(ValueOpcode::ShiftRightLogical64, {Value(uint64_t{0x10000000000ull}), Value(40u)});
  const auto shr64_id = fixture.Emit(ValueOpcode::ShiftRightLogical64, {sym64, Value(0u)});
  const auto sra64 = fixture.Emit(ValueOpcode::ShiftRightArithmetic64, {Value(static_cast<uint64_t>(-16ll)), Value(2u)});
  // WqmU64
  const auto wqm = fixture.Emit(ValueOpcode::WqmU64, {Value(uint64_t{1ull})});
  // BitCount32 / BitCount64
  const auto pop32 = fixture.Emit(ValueOpcode::BitCount32, {Value(0x10101010u)});
  const auto pop64 = fixture.Emit(ValueOpcode::BitCount64, {Value(uint64_t{0x1010101010101010ull})});
  // SMin32 / SMax32 / UMin32 / UMax32
  const auto smin = fixture.Emit(ValueOpcode::SMin32, {Value(static_cast<uint32_t>(-10)), Value(5u)});
  const auto smax = fixture.Emit(ValueOpcode::SMax32, {Value(static_cast<uint32_t>(-10)), Value(5u)});
  const auto umin = fixture.Emit(ValueOpcode::UMin32, {Value(10u), Value(50u)});
  const auto umax = fixture.Emit(ValueOpcode::UMax32, {Value(10u), Value(50u)});
  // BitwiseAnd64
  const auto and64 = fixture.Emit(ValueOpcode::BitwiseAnd64, {Value(uint64_t{0xff00ff00ff00ff00ull}), Value(uint64_t{0x0f0f0f0f0f0f0f0full})});
  const auto and64_id = fixture.Emit(ValueOpcode::BitwiseAnd64, {sym64, Value(UINT64_MAX)});
  // BitwiseNot32
  const auto not32 = fixture.Emit(ValueOpcode::BitwiseNot32, {Value(0x0000ffffu)});

  ConstantPropagationPass(fixture.program.blocks);

  Check(sub32.Resolve() == Value(42u), "ISub32 constant fold failed");
  Check(sub32_id.Resolve() == sym32, "ISub32 zero rhs identity failed");
  Check(sub64.Resolve() == Value(uint64_t{58ull}), "ISub64 constant fold failed");
  Check(sub64_id.Resolve() == sym64, "ISub64 zero rhs identity failed");
  Check(mul64.Resolve() == Value(uint64_t{42ull}), "IMul64 constant fold failed");
  Check(mul64_id.Resolve() == sym64, "IMul64 one rhs identity failed");
  Check(div32.Resolve() == Value(42u), "UDiv32 constant fold failed");
  Check(div32_id.Resolve() == Value(42u), "UDiv32 one rhs identity failed");
  Check(smulhi.Resolve() == Value(0x1000000u), "SMulHi constant fold failed");
  Check(umulhi.Resolve() == Value(1u), "UMulHi constant fold failed");
  Check(abs_pos.Resolve() == Value(42u), "IAbs32 positive fold failed");
  Check(abs_neg.Resolve() == Value(0x7ffffffbu), "IAbs32 negative fold failed");
  Check(shl32.Resolve() == Value(16u), "ShiftLeftLogical32 fold failed");
  Check(shl32_id.Resolve() == sym32, "ShiftLeftLogical32 zero shift identity failed");
  Check(shr32.Resolve() == Value(8u), "ShiftRightLogical32 fold failed");
  Check(shr32_id.Resolve() == sym32, "ShiftRightLogical32 zero shift identity failed");
  Check(sra32.Resolve() == Value(static_cast<uint32_t>(-4)), "ShiftRightArithmetic32 fold failed");
  Check(shl64.Resolve() == Value(uint64_t{0x10000000000ull}), "ShiftLeftLogical64 fold failed");
  Check(shl64_id.Resolve() == sym64, "ShiftLeftLogical64 zero shift identity failed");
  Check(shr64.Resolve() == Value(uint64_t{1ull}), "ShiftRightLogical64 fold failed");
  Check(shr64_id.Resolve() == sym64, "ShiftRightLogical64 zero shift identity failed");
  Check(sra64.Resolve() == Value(static_cast<uint64_t>(-4ll)), "ShiftRightArithmetic64 fold failed");
  Check(wqm.Resolve() == Value(uint64_t{0xfull}), "WqmU64 fold failed");
  Check(pop32.Resolve() == Value(4u), "BitCount32 fold failed");
  Check(pop64.Resolve() == Value(8u), "BitCount64 fold failed");
  Check(smin.Resolve() == Value(static_cast<uint32_t>(-10)), "SMin32 fold failed");
  Check(smax.Resolve() == Value(5u), "SMax32 fold failed");
  Check(umin.Resolve() == Value(10u), "UMin32 fold failed");
  Check(umax.Resolve() == Value(50u), "UMax32 fold failed");
  Check(and64.Resolve() == Value(uint64_t{0x0f000f000f000f00ull}), "BitwiseAnd64 fold failed");
  Check(and64_id.Resolve() == sym64, "BitwiseAnd64 identity failed");
  Check(not32.Resolve() == Value(0xffff0000u), "BitwiseNot32 fold failed");
}

void TestConstantPropagationComparisonsAndLogical() {
  Fixture fixture;
  // 32-bit comparison
  const auto ult = fixture.Emit(ValueOpcode::ULessThan32, {Value(10u), Value(20u)});
  const auto ulte = fixture.Emit(ValueOpcode::ULessThanEqual32, {Value(10u), Value(10u)});
  const auto ugt = fixture.Emit(ValueOpcode::UGreaterThan32, {Value(20u), Value(10u)});
  const auto ugte = fixture.Emit(ValueOpcode::UGreaterThanEqual32, {Value(10u), Value(10u)});
  const auto slt = fixture.Emit(ValueOpcode::SLessThan32, {Value(static_cast<uint32_t>(-5)), Value(5u)});
  const auto slte = fixture.Emit(ValueOpcode::SLessThanEqual32, {Value(5u), Value(5u)});
  const auto sgt = fixture.Emit(ValueOpcode::SGreaterThan32, {Value(5u), Value(static_cast<uint32_t>(-5))});
  const auto sgte = fixture.Emit(ValueOpcode::SGreaterThanEqual32, {Value(5u), Value(5u)});
  const auto ieq32 = fixture.Emit(ValueOpcode::IEqual32, {Value(42u), Value(42u)});
  const auto ineq32 = fixture.Emit(ValueOpcode::INotEqual32, {Value(42u), Value(42u)});

  // 64-bit comparison
  const auto ieq64 = fixture.Emit(ValueOpcode::IEqual64, {Value(uint64_t{100ull}), Value(uint64_t{100ull})});
  const auto ineq64 = fixture.Emit(ValueOpcode::INotEqual64, {Value(uint64_t{100ull}), Value(uint64_t{200ull})});
  const auto ult64 = fixture.Emit(ValueOpcode::ULessThan64, {Value(uint64_t{100ull}), Value(uint64_t{200ull})});
  const auto ugt64 = fixture.Emit(ValueOpcode::UGreaterThan64, {Value(uint64_t{200ull}), Value(uint64_t{100ull})});
  const auto slt64 = fixture.Emit(ValueOpcode::SLessThan64, {Value(static_cast<uint64_t>(-10ll)), Value(uint64_t{10ull})});

  // Logical operations
  const auto land_const = fixture.Emit(ValueOpcode::LogicalAnd, {Value(true), Value(false)});
  const auto lor_const = fixture.Emit(ValueOpcode::LogicalOr, {Value(true), Value(false)});
  const auto lxor_const = fixture.Emit(ValueOpcode::LogicalXor, {Value(true), Value(false)});
  const auto lnot_true = fixture.Emit(ValueOpcode::LogicalNot, {Value(true)});
  const auto lnot_false = fixture.Emit(ValueOpcode::LogicalNot, {Value(false)});

  // Identity / short-circuit with symbolic argument
  const auto sym_cond = fixture.Emit(ValueOpcode::UndefU1);
  const auto land_short = fixture.Emit(ValueOpcode::LogicalAnd, {Value(false), sym_cond});
  const auto land_id = fixture.Emit(ValueOpcode::LogicalAnd, {Value(true), sym_cond});
  const auto lor_short = fixture.Emit(ValueOpcode::LogicalOr, {Value(true), sym_cond});
  const auto lor_id = fixture.Emit(ValueOpcode::LogicalOr, {Value(false), sym_cond});
  const auto lxor_id = fixture.Emit(ValueOpcode::LogicalXor, {Value(false), sym_cond});
  // Double logical not
  const auto lnot1 = fixture.Emit(ValueOpcode::LogicalNot, {sym_cond});
  const auto lnot2 = fixture.Emit(ValueOpcode::LogicalNot, {lnot1});

  ConstantPropagationPass(fixture.program.blocks);

  Check(ult.Resolve() == Value(true), "ULessThan32 fold failed");
  Check(ulte.Resolve() == Value(true), "ULessThanEqual32 fold failed");
  Check(ugt.Resolve() == Value(true), "UGreaterThan32 fold failed");
  Check(ugte.Resolve() == Value(true), "UGreaterThanEqual32 fold failed");
  Check(slt.Resolve() == Value(true), "SLessThan32 fold failed");
  Check(slte.Resolve() == Value(true), "SLessThanEqual32 fold failed");
  Check(sgt.Resolve() == Value(true), "SGreaterThan32 fold failed");
  Check(sgte.Resolve() == Value(true), "SGreaterThanEqual32 fold failed");
  Check(ieq32.Resolve() == Value(true), "IEqual32 fold failed");
  Check(ineq32.Resolve() == Value(false), "INotEqual32 fold failed");

  Check(ieq64.Resolve() == Value(true), "IEqual64 fold failed");
  Check(ineq64.Resolve() == Value(true), "INotEqual64 fold failed");
  Check(ult64.Resolve() == Value(true), "ULessThan64 fold failed");
  Check(ugt64.Resolve() == Value(true), "UGreaterThan64 fold failed");
  Check(slt64.Resolve() == Value(true), "SLessThan64 fold failed");

  Check(land_const.Resolve() == Value(false), "LogicalAnd const fold failed");
  Check(lor_const.Resolve() == Value(true), "LogicalOr const fold failed");
  Check(lxor_const.Resolve() == Value(true), "LogicalXor const fold failed");
  Check(lnot_true.Resolve() == Value(false), "LogicalNot true fold failed");
  Check(lnot_false.Resolve() == Value(true), "LogicalNot false fold failed");

  Check(land_short.Resolve() == Value(false), "LogicalAnd short-circuit failed");
  Check(land_id.Resolve() == sym_cond, "LogicalAnd identity failed");
  Check(lor_short.Resolve() == Value(true), "LogicalOr short-circuit failed");
  Check(lor_id.Resolve() == sym_cond, "LogicalOr identity failed");
  Check(lxor_id.Resolve() == sym_cond, "LogicalXor identity failed");
  Check(lnot2.Resolve() == sym_cond, "LogicalNot cancellation failed");
}

void TestConstantPropagationConversionsAndCasts() {
  Fixture fixture;
  // Select
  const auto sel_t = fixture.Emit(ValueOpcode::SelectU32, {Value(true), Value(42u), Value(99u)});
  const auto sel_f = fixture.Emit(ValueOpcode::SelectU32, {Value(false), Value(42u), Value(99u)});
  const auto undef_cond = fixture.Emit(ValueOpcode::UndefU1);
  const auto sel_same = fixture.Emit(ValueOpcode::SelectU32, {undef_cond, Value(77u), Value(77u)});

  // BitCast F32 <-> U32
  const auto cast_u_f = fixture.Emit(ValueOpcode::BitCastU32F32, {Value::F32(1.0f)});
  const auto cast_f_u = fixture.Emit(ValueOpcode::BitCastF32U32, {Value(0x3f800000u)});
  // BitCast cancellation
  const auto undef_u32 = fixture.Emit(ValueOpcode::UndefU32);
  const auto f_from_u = fixture.Emit(ValueOpcode::BitCastF32U32, {undef_u32});
  const auto u_from_f = fixture.Emit(ValueOpcode::BitCastU32F32, {f_from_u});

  // Convert U16/U32
  const auto c_u16_u32 = fixture.Emit(ValueOpcode::ConvertU16U32, {Value(0x12345678u)});
  const auto c_u32_u16 = fixture.Emit(ValueOpcode::ConvertU32U16, {Value(static_cast<uint16_t>(0x5678u))});
  const auto c_u16_cancel = fixture.Emit(ValueOpcode::ConvertU16U32, {undef_u32});
  const auto c_u32_cancel = fixture.Emit(ValueOpcode::ConvertU32U16, {c_u16_cancel});

  // Convert U8/U32
  const auto c_u8_u32 = fixture.Emit(ValueOpcode::ConvertU8U32, {Value(0x12345678u)});
  const auto c_u32_u8 = fixture.Emit(ValueOpcode::ConvertU32U8, {Value(static_cast<uint8_t>(0x78u))});
  const auto c_u8_cancel = fixture.Emit(ValueOpcode::ConvertU8U32, {undef_u32});
  const auto c_u32_cancel8 = fixture.Emit(ValueOpcode::ConvertU32U8, {c_u8_cancel});

  // Composite Construct / Extract
  const auto comp_u64 = fixture.Emit(ValueOpcode::CompositeConstructU64, {Value(0x12345678u), Value(0x9abcdef0u)});
  const auto ext_u64_0 = fixture.Emit(ValueOpcode::CompositeExtractU64, {comp_u64, Value(0u)});
  const auto ext_u64_1 = fixture.Emit(ValueOpcode::CompositeExtractU64, {comp_u64, Value(1u)});

  const auto comp3 = fixture.Emit(ValueOpcode::CompositeConstructU32x3, {Value(10u), Value(20u), Value(30u)});
  const auto ext3_1 = fixture.Emit(ValueOpcode::CompositeExtractU32x3, {comp3, Value(1u)});

  const auto comp4 = fixture.Emit(ValueOpcode::CompositeConstructU32x4, {Value(1u), Value(2u), Value(3u), Value(4u)});
  const auto ext4_3 = fixture.Emit(ValueOpcode::CompositeExtractU32x4, {comp4, Value(3u)});

  // BitFieldInsert zero count identity
  const auto bfi_zero = fixture.Emit(ValueOpcode::BitFieldInsert, {Value(0x12345678u), Value(0x99u), Value(8u), Value(0u)});

  ConstantPropagationPass(fixture.program.blocks);

  Check(sel_t.Resolve() == Value(42u), "Select true fold failed");
  Check(sel_f.Resolve() == Value(99u), "Select false fold failed");
  Check(sel_same.Resolve() == Value(77u), "Select same values fold failed");

  Check(cast_u_f.Resolve() == Value(0x3f800000u), "BitCastU32F32 fold failed");
  Check(cast_f_u.Resolve() == Value::F32(1.0f), "BitCastF32U32 fold failed");
  Check(u_from_f.Resolve() == undef_u32, "BitCast round-trip cancellation failed");

  Check(c_u16_u32.Resolve() == Value(static_cast<uint16_t>(0x5678u)), "ConvertU16U32 fold failed");
  Check(c_u32_u16.Resolve() == Value(0x5678u), "ConvertU32U16 fold failed");
  Check(c_u32_cancel.Resolve() == undef_u32, "ConvertU32U16 round-trip cancellation failed");

  Check(c_u8_u32.Resolve() == Value(static_cast<uint8_t>(0x78u)), "ConvertU8U32 fold failed");
  Check(c_u32_u8.Resolve() == Value(0x78u), "ConvertU32U8 fold failed");
  Check(c_u32_cancel8.Resolve() == undef_u32, "ConvertU32U8 round-trip cancellation failed");

  Check(comp_u64.Resolve() == Value(uint64_t{0x9abcdef012345678ull}), "CompositeConstructU64 fold failed");
  Check(ext_u64_0.Resolve() == Value(0x12345678u), "CompositeExtractU64 component 0 fold failed");
  Check(ext_u64_1.Resolve() == Value(0x9abcdef0u), "CompositeExtractU64 component 1 fold failed");

  Check(ext3_1.Resolve() == Value(20u), "CompositeExtractU32x3 component 1 fold failed");
  Check(ext4_3.Resolve() == Value(4u), "CompositeExtractU32x4 component 3 fold failed");
  Check(bfi_zero.Resolve() == Value(0x12345678u), "BitFieldInsert zero count identity failed");
}

void TestDeadCodeEliminationChainsAndSideEffects() {
  Fixture fixture;
  // Multistep unused dead code chain:
  // v1 = IAdd32(1, 2)
  // v2 = IMul32(v1, 3)
  // v3 = ISub32(v2, 4)
  // (v3 has no uses, so v3, v2, v1 should all be dead)
  const auto v1 = fixture.Emit(ValueOpcode::IAdd32, {Value(1u), Value(2u)});
  const auto v2 = fixture.Emit(ValueOpcode::IMul32, {v1, Value(3u)});
  fixture.Emit(ValueOpcode::ISub32, {v2, Value(4u)});

  // Side-effecting instruction with no uses: Barrier
  // Must NOT be eliminated!
  fixture.Emit(ValueOpcode::Barrier);

  // Identity chaining:
  // id1 = Identity(live_const)
  // id2 = Identity(id1)
  // ref = ReferenceU32(id2)
  const auto live_const = Value(1234u);
  const auto id1 = fixture.Emit(ValueOpcode::Identity, {live_const});
  const auto id2 = fixture.Emit(ValueOpcode::Identity, {id1});
  fixture.Emit(ValueOpcode::ReferenceU32, {id2});

  RemoveIdentities(fixture.program.blocks);
  EliminateDeadCode(fixture.program.blocks);

  const auto &instructions = fixture.BlockAt().Instructions();
  // We expect only 2 instructions to remain: Barrier and ReferenceU32(1234u)
  Check(instructions.size() == 2, "DeadCodeElimination did not remove complete dead chain or kept identities");
  auto it = instructions.begin();
  Check(it->GetOpcode() == ValueOpcode::Barrier, "Barrier instruction was wrongly eliminated");
  ++it;
  Check(it->GetOpcode() == ValueOpcode::ReferenceU32 && it->Arg(0).Resolve() == live_const,
        "RemoveIdentities failed to resolve transitive identity to live constant");
}

} // namespace

namespace Common {

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

int DbgExitHandler(const char *, int, std::string_view text) {
  throw std::runtime_error(std::string(text));
}

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view text) {
  throw std::runtime_error(std::string(text));
}

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  try {
    TestImmediateFlatteningAndGvn();
    TestRawScalarComponentAlignment();
    TestScalarMemoryDomainMismatchFails();
    TestDynamicReadRemainsTyped();
    TestNestedSrtWalk();
    TestShaderBaseAndUserData();
    TestCarryAndBitFields();
    TestInvariantAndDivergentPhi();
    TestOperandMutationLifecycle();
    TestControlDependentStandaloneLoadStaysTyped();
    TestRuntime64BitDescriptorOps();
    TestUniformFirstLaneSamplerLod();
    TestFloatComparisonDescriptorInputs();
    TestSharedIntegerRuntimeDependencies();
    TestConstantBufferBounds();
    TestReadLaneElimination();
    TestOptimizationPipeline();
    TestControlFlowValueSurvivesReadLaneFolding();
    TestDeadPhiCyclesAndPlanningRoots();
    TestUndefinedRuntimeValueFails();
    TestConstantPropagationArithmeticAndBitwise();
    TestConstantPropagationComparisonsAndLogical();
    TestConstantPropagationConversionsAndCasts();
    TestDeadCodeEliminationChainsAndSideEffects();
    std::cout << "TypedValuePlanningTests: all cases passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "TypedValuePlanningTests: failed: " << e.what() << '\n';
    return 1;
  }
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Program.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/ConstantPropagation.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/DeadCodeElimination.cpp"


