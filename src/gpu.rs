// Vulkan GPU: compute pipelines + graphics rendering pipeline
// dlopen libvulkan.so.1, create compute/graphics pipelines, offscreen render + readback

#![allow(unsafe_op_in_unsafe_fn, non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};
use std::ptr;

use crate::gpu_kernels;
use crate::gpu_shaders;
// math types used via GpuPushConstants

// --- libc ---
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_LAZY: c_int = 1;

// --- Byte casting for GPU buffer upload/download ---
pub fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
pub fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

// --- Vulkan handle types ---
type VkInstance = *mut c_void;
type VkPhysicalDevice = *mut c_void;
type VkDevice = *mut c_void;
type VkQueue = *mut c_void;
type VkCommandBuffer = *mut c_void;
type VkBuffer = u64;
type VkDeviceMemory = u64;
type VkShaderModule = u64;
type VkDescriptorSetLayout = u64;
type VkPipelineLayout = u64;
type VkPipeline = u64;
type VkDescriptorPool = u64;
type VkDescriptorSet = u64;
type VkCommandPool = u64;
type VkFence = u64;
type VkDeviceSize = u64;
type VkImage = u64;
type VkImageView = u64;
type VkRenderPass = u64;
type VkFramebufferVk = u64; // avoid clash with our Framebuffer

// --- Vulkan constants ---
const VK_SUCCESS: i32 = 0;
const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_FENCE_CREATE_SIGNALED_BIT: u32 = 1;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: u32 = 44;

const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x02;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x04;
const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x08;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x20;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 1;
const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x800;
const VK_ACCESS_SHADER_READ_BIT: u32 = 0x20;
const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x40;
const VK_WHOLE_SIZE: u64 = u64::MAX;
const VK_QUEUE_FAMILY_IGNORED: u32 = u32::MAX;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 2;

const VK_API_VERSION_1_1: u32 = (1 << 22) | (1 << 12);

// --- Graphics pipeline constants ---
const VK_QUEUE_GRAPHICS_BIT: u32 = 1;

const VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO: u32 = 38;
const VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO: u32 = 37;
const VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: u32 = 28;
const VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO: u32 = 14;
const VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO: u32 = 15;
const VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: u32 = 19;
const VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: u32 = 20;
const VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: u32 = 22;
const VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: u32 = 23;
const VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: u32 = 24;
const VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO: u32 = 25;
const VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: u32 = 26;
const VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: u32 = 27;
const VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO: u32 = 43;
const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER: u32 = 45;

// Formats
const VK_FORMAT_R32G32B32_SFLOAT: u32 = 106;
const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
const VK_FORMAT_D32_SFLOAT: u32 = 126;

// Image
const VK_IMAGE_TYPE_2D: u32 = 1;
const VK_IMAGE_TILING_OPTIMAL: u32 = 0;
const VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: u32 = 0x10;
const VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT: u32 = 0x20;
const VK_IMAGE_USAGE_TRANSFER_SRC_BIT: u32 = 0x01;
const VK_SAMPLE_COUNT_1_BIT: u32 = 1;
const VK_IMAGE_VIEW_TYPE_2D: u32 = 1;
const VK_IMAGE_ASPECT_COLOR_BIT: u32 = 0x01;
const VK_IMAGE_ASPECT_DEPTH_BIT: u32 = 0x02;
const VK_COMPONENT_SWIZZLE_IDENTITY: u32 = 0;

// Image layouts
const VK_IMAGE_LAYOUT_UNDEFINED: u32 = 0;
const VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: u32 = 2;
const VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: u32 = 3;
const VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: u32 = 6;

// Buffer usage
const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: u32 = 0x80;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x02;

// Pipeline
const VK_SHADER_STAGE_VERTEX_BIT: u32 = 0x01;
const VK_SHADER_STAGE_FRAGMENT_BIT: u32 = 0x10;
const VK_PIPELINE_BIND_POINT_GRAPHICS: u32 = 0;
const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: u32 = 3;
const VK_POLYGON_MODE_FILL: u32 = 0;
const VK_CULL_MODE_NONE: u32 = 0;
const _VK_FRONT_FACE_COUNTER_CLOCKWISE: u32 = 0;
const _VK_FRONT_FACE_CLOCKWISE: u32 = 1;
const VK_COMPARE_OP_LESS: u32 = 1;
const VK_DYNAMIC_STATE_VIEWPORT: u32 = 0;
const VK_DYNAMIC_STATE_SCISSOR: u32 = 1;
const VK_LOGIC_OP_COPY: u32 = 3;

// Subpass/attachment
const VK_SUBPASS_EXTERNAL: u32 = u32::MAX;
const VK_ATTACHMENT_LOAD_OP_CLEAR: u32 = 1;
const VK_ATTACHMENT_LOAD_OP_DONT_CARE: u32 = 2;
const VK_ATTACHMENT_STORE_OP_STORE: u32 = 0;
const VK_ATTACHMENT_STORE_OP_DONT_CARE: u32 = 1;
const VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT: u32 = 0x400;
const VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT: u32 = 0x100;
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x1000;
const VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: u32 = 0x100;
const VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT: u32 = 0x400;
const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x800;
const VK_ACCESS_HOST_READ_BIT: u32 = 0x2000;
const VK_PIPELINE_STAGE_HOST_BIT: u32 = 0x4000;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x01;
const VK_SUBPASS_CONTENTS_INLINE: u32 = 0;

// --- Vulkan structs ---

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    app_name: *const c_char,
    app_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    app_info: *const VkApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
    features: *const c_void,
}

#[repr(C)]
#[derive(Clone)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
struct VkMemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: VkDeviceSize,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: VkDeviceSize,
    memory_type_index: u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    p_code: *const u32,
}

#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    p_immutable_samplers: *const u64,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
struct VkPushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    p_name: *const c_char,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
struct VkDescriptorPoolSize {
    typ: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: VkDeviceSize,
    range: VkDeviceSize,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const c_void,
    p_buffer_info: *const VkDescriptorBufferInfo,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const u64,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct VkBufferMemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: u32,
    dst_access_mask: u32,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    buffer: VkBuffer,
    offset: VkDeviceSize,
    size: VkDeviceSize,
}

// --- Graphics pipeline structs ---

#[repr(C)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
struct VkImageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    image_type: u32,
    format: u32,
    extent: VkExtent3D,
    mip_levels: u32,
    array_layers: u32,
    samples: u32,
    tiling: u32,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
    initial_layout: u32,
}

#[repr(C)]
struct VkComponentMapping {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

#[repr(C)]
struct VkImageSubresourceRange {
    aspect_mask: u32,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
}

#[repr(C)]
struct VkImageViewCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    image: VkImage,
    view_type: u32,
    format: u32,
    components: VkComponentMapping,
    subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
struct VkAttachmentDescription {
    flags: u32,
    format: u32,
    samples: u32,
    load_op: u32,
    store_op: u32,
    stencil_load_op: u32,
    stencil_store_op: u32,
    initial_layout: u32,
    final_layout: u32,
}

#[repr(C)]
struct VkAttachmentReference {
    attachment: u32,
    layout: u32,
}

#[repr(C)]
struct VkSubpassDescription {
    flags: u32,
    pipeline_bind_point: u32,
    input_attachment_count: u32,
    p_input_attachments: *const VkAttachmentReference,
    color_attachment_count: u32,
    p_color_attachments: *const VkAttachmentReference,
    p_resolve_attachments: *const VkAttachmentReference,
    p_depth_stencil_attachment: *const VkAttachmentReference,
    preserve_attachment_count: u32,
    p_preserve_attachments: *const u32,
}

#[repr(C)]
struct VkSubpassDependency {
    src_subpass: u32,
    dst_subpass: u32,
    src_stage_mask: u32,
    dst_stage_mask: u32,
    src_access_mask: u32,
    dst_access_mask: u32,
    dependency_flags: u32,
}

#[repr(C)]
struct VkRenderPassCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    attachment_count: u32,
    p_attachments: *const VkAttachmentDescription,
    subpass_count: u32,
    p_subpasses: *const VkSubpassDescription,
    dependency_count: u32,
    p_dependencies: *const VkSubpassDependency,
}

#[repr(C)]
struct VkFramebufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    render_pass: VkRenderPass,
    attachment_count: u32,
    p_attachments: *const VkImageView,
    width: u32,
    height: u32,
    layers: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkClearColorValue {
    float32: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkClearDepthStencilValue {
    depth: f32,
    stencil: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
union VkClearValue {
    color: VkClearColorValue,
    depth_stencil: VkClearDepthStencilValue,
}

#[repr(C)]
struct VkRect2D {
    offset_x: i32,
    offset_y: i32,
    extent_w: u32,
    extent_h: u32,
}

#[repr(C)]
struct VkRenderPassBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    render_pass: VkRenderPass,
    framebuffer: VkFramebufferVk,
    render_area: VkRect2D,
    clear_value_count: u32,
    p_clear_values: *const VkClearValue,
}

#[repr(C)]
struct VkVertexInputBindingDescription {
    binding: u32,
    stride: u32,
    input_rate: u32,
}

#[repr(C)]
struct VkVertexInputAttributeDescription {
    location: u32,
    binding: u32,
    format: u32,
    offset: u32,
}

#[repr(C)]
struct VkPipelineVertexInputStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    vertex_binding_description_count: u32,
    p_vertex_binding_descriptions: *const VkVertexInputBindingDescription,
    vertex_attribute_description_count: u32,
    p_vertex_attribute_descriptions: *const VkVertexInputAttributeDescription,
}

#[repr(C)]
struct VkPipelineInputAssemblyStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    topology: u32,
    primitive_restart_enable: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkViewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

#[repr(C)]
struct VkPipelineViewportStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    viewport_count: u32,
    p_viewports: *const VkViewport,
    scissor_count: u32,
    p_scissors: *const VkRect2D,
}

#[repr(C)]
struct VkPipelineRasterizationStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    depth_clamp_enable: u32,
    rasterizer_discard_enable: u32,
    polygon_mode: u32,
    cull_mode: u32,
    front_face: u32,
    depth_bias_enable: u32,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,
}

#[repr(C)]
struct VkPipelineMultisampleStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    rasterization_samples: u32,
    sample_shading_enable: u32,
    min_sample_shading: f32,
    p_sample_mask: *const u32,
    alpha_to_coverage_enable: u32,
    alpha_to_one_enable: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkStencilOpState {
    fail_op: u32,
    pass_op: u32,
    depth_fail_op: u32,
    compare_op: u32,
    compare_mask: u32,
    write_mask: u32,
    reference: u32,
}

#[repr(C)]
struct VkPipelineDepthStencilStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    depth_test_enable: u32,
    depth_write_enable: u32,
    depth_compare_op: u32,
    depth_bounds_test_enable: u32,
    stencil_test_enable: u32,
    front: VkStencilOpState,
    back: VkStencilOpState,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
}

#[repr(C)]
struct VkPipelineColorBlendAttachmentState {
    blend_enable: u32,
    src_color_blend_factor: u32,
    dst_color_blend_factor: u32,
    color_blend_op: u32,
    src_alpha_blend_factor: u32,
    dst_alpha_blend_factor: u32,
    alpha_blend_op: u32,
    color_write_mask: u32,
}

#[repr(C)]
struct VkPipelineColorBlendStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    logic_op_enable: u32,
    logic_op: u32,
    attachment_count: u32,
    p_attachments: *const VkPipelineColorBlendAttachmentState,
    blend_constants: [f32; 4],
}

#[repr(C)]
struct VkPipelineDynamicStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    dynamic_state_count: u32,
    p_dynamic_states: *const u32,
}

#[repr(C)]
struct VkGraphicsPipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage_count: u32,
    p_stages: *const VkPipelineShaderStageCreateInfo,
    p_vertex_input_state: *const VkPipelineVertexInputStateCreateInfo,
    p_input_assembly_state: *const VkPipelineInputAssemblyStateCreateInfo,
    p_tessellation_state: *const c_void,
    p_viewport_state: *const VkPipelineViewportStateCreateInfo,
    p_rasterization_state: *const VkPipelineRasterizationStateCreateInfo,
    p_multisample_state: *const VkPipelineMultisampleStateCreateInfo,
    p_depth_stencil_state: *const VkPipelineDepthStencilStateCreateInfo,
    p_color_blend_state: *const VkPipelineColorBlendStateCreateInfo,
    p_dynamic_state: *const VkPipelineDynamicStateCreateInfo,
    layout: VkPipelineLayout,
    render_pass: VkRenderPass,
    subpass: u32,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
struct VkImageMemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: u32,
    dst_access_mask: u32,
    old_layout: u32,
    new_layout: u32,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: VkImage,
    subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
struct VkBufferImageCopy {
    buffer_offset: VkDeviceSize,
    buffer_row_length: u32,
    buffer_image_height: u32,
    image_subresource_layers: VkImageSubresourceLayers,
    image_offset_x: i32,
    image_offset_y: i32,
    image_offset_z: i32,
    image_extent: VkExtent3D,
}

#[repr(C)]
struct VkImageSubresourceLayers {
    aspect_mask: u32,
    mip_level: u32,
    base_array_layer: u32,
    layer_count: u32,
}

// --- Function pointer types ---
type FnGetInstanceProcAddr = unsafe extern "C" fn(VkInstance, *const c_char) -> *mut c_void;
type FnCreateInstance = unsafe extern "C" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> i32;
type FnDestroyInstance = unsafe extern "C" fn(VkInstance, *const c_void);
type FnEnumPhysDevices = unsafe extern "C" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> i32;
type FnGetPhysDevProps = unsafe extern "C" fn(VkPhysicalDevice, *mut [u8; 1024]);
type FnGetPhysDevMemProps = unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
type FnGetPhysDevQueueFamProps = unsafe extern "C" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
type FnCreateDevice = unsafe extern "C" fn(VkPhysicalDevice, *const VkDeviceCreateInfo, *const c_void, *mut VkDevice) -> i32;
type FnDestroyDevice = unsafe extern "C" fn(VkDevice, *const c_void);
type FnGetDeviceQueue = unsafe extern "C" fn(VkDevice, u32, u32, *mut VkQueue);
type FnCreateBuffer = unsafe extern "C" fn(VkDevice, *const VkBufferCreateInfo, *const c_void, *mut VkBuffer) -> i32;
type FnDestroyBuffer = unsafe extern "C" fn(VkDevice, VkBuffer, *const c_void);
type FnGetBufMemReqs = unsafe extern "C" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
type FnAllocMem = unsafe extern "C" fn(VkDevice, *const VkMemoryAllocateInfo, *const c_void, *mut VkDeviceMemory) -> i32;
type FnFreeMem = unsafe extern "C" fn(VkDevice, VkDeviceMemory, *const c_void);
type FnBindBufMem = unsafe extern "C" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> i32;
type FnMapMem = unsafe extern "C" fn(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, u32, *mut *mut c_void) -> i32;
type FnUnmapMem = unsafe extern "C" fn(VkDevice, VkDeviceMemory);
type FnCreateShaderModule = unsafe extern "C" fn(VkDevice, *const VkShaderModuleCreateInfo, *const c_void, *mut VkShaderModule) -> i32;
type FnDestroyShaderModule = unsafe extern "C" fn(VkDevice, VkShaderModule, *const c_void);
type FnCreateDescSetLayout = unsafe extern "C" fn(VkDevice, *const VkDescriptorSetLayoutCreateInfo, *const c_void, *mut VkDescriptorSetLayout) -> i32;
type FnDestroyDescSetLayout = unsafe extern "C" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
type FnCreatePipelineLayout = unsafe extern "C" fn(VkDevice, *const VkPipelineLayoutCreateInfo, *const c_void, *mut VkPipelineLayout) -> i32;
type FnDestroyPipelineLayout = unsafe extern "C" fn(VkDevice, VkPipelineLayout, *const c_void);
type FnCreateComputePipelines = unsafe extern "C" fn(VkDevice, u64, u32, *const VkComputePipelineCreateInfo, *const c_void, *mut VkPipeline) -> i32;
type FnDestroyPipeline = unsafe extern "C" fn(VkDevice, VkPipeline, *const c_void);
type FnCreateDescPool = unsafe extern "C" fn(VkDevice, *const VkDescriptorPoolCreateInfo, *const c_void, *mut VkDescriptorPool) -> i32;
type FnDestroyDescPool = unsafe extern "C" fn(VkDevice, VkDescriptorPool, *const c_void);
type FnResetDescPool = unsafe extern "C" fn(VkDevice, VkDescriptorPool, u32) -> i32;
type FnAllocDescSets = unsafe extern "C" fn(VkDevice, *const VkDescriptorSetAllocateInfo, *mut VkDescriptorSet) -> i32;
type FnUpdateDescSets = unsafe extern "C" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void);
type FnCreateCmdPool = unsafe extern "C" fn(VkDevice, *const VkCommandPoolCreateInfo, *const c_void, *mut VkCommandPool) -> i32;
type FnDestroyCmdPool = unsafe extern "C" fn(VkDevice, VkCommandPool, *const c_void);
type FnAllocCmdBufs = unsafe extern "C" fn(VkDevice, *const VkCommandBufferAllocateInfo, *mut VkCommandBuffer) -> i32;
type FnBeginCmdBuf = unsafe extern "C" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> i32;
type FnEndCmdBuf = unsafe extern "C" fn(VkCommandBuffer) -> i32;
type FnCmdBindPipeline = unsafe extern "C" fn(VkCommandBuffer, u32, VkPipeline);
type FnCmdBindDescSets = unsafe extern "C" fn(VkCommandBuffer, u32, VkPipelineLayout, u32, u32, *const VkDescriptorSet, u32, *const u32);
type FnCmdDispatch = unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32);
type FnCmdPipelineBarrier = unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32, u32, *const c_void, u32, *const VkBufferMemoryBarrier, u32, *const c_void);
type FnCmdPushConstants = unsafe extern "C" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const c_void);
type FnCreateFence = unsafe extern "C" fn(VkDevice, *const VkFenceCreateInfo, *const c_void, *mut VkFence) -> i32;
type FnDestroyFence = unsafe extern "C" fn(VkDevice, VkFence, *const c_void);
type FnResetFences = unsafe extern "C" fn(VkDevice, u32, *const VkFence) -> i32;
type FnWaitForFences = unsafe extern "C" fn(VkDevice, u32, *const VkFence, u32, u64) -> i32;
type FnQueueSubmit = unsafe extern "C" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> i32;

// Graphics function pointer types
type FnCreateImage = unsafe extern "C" fn(VkDevice, *const VkImageCreateInfo, *const c_void, *mut VkImage) -> i32;
type FnDestroyImage = unsafe extern "C" fn(VkDevice, VkImage, *const c_void);
type FnGetImageMemReqs = unsafe extern "C" fn(VkDevice, VkImage, *mut VkMemoryRequirements);
type FnBindImageMem = unsafe extern "C" fn(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) -> i32;
type FnCreateImageView = unsafe extern "C" fn(VkDevice, *const VkImageViewCreateInfo, *const c_void, *mut VkImageView) -> i32;
type FnDestroyImageView = unsafe extern "C" fn(VkDevice, VkImageView, *const c_void);
type FnCreateRenderPass = unsafe extern "C" fn(VkDevice, *const VkRenderPassCreateInfo, *const c_void, *mut VkRenderPass) -> i32;
type FnDestroyRenderPass = unsafe extern "C" fn(VkDevice, VkRenderPass, *const c_void);
type FnCreateFramebuffer = unsafe extern "C" fn(VkDevice, *const VkFramebufferCreateInfo, *const c_void, *mut VkFramebufferVk) -> i32;
type FnDestroyFramebuffer = unsafe extern "C" fn(VkDevice, VkFramebufferVk, *const c_void);
type FnCreateGraphicsPipelines = unsafe extern "C" fn(VkDevice, u64, u32, *const VkGraphicsPipelineCreateInfo, *const c_void, *mut VkPipeline) -> i32;
type FnCmdBeginRenderPass = unsafe extern "C" fn(VkCommandBuffer, *const VkRenderPassBeginInfo, u32);
type FnCmdEndRenderPass = unsafe extern "C" fn(VkCommandBuffer);
type FnCmdBindVertexBuffers = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkBuffer, *const VkDeviceSize);
type FnCmdDraw = unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32, u32);
type FnCmdSetViewport = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkViewport);
type FnCmdSetScissor = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkRect2D);
type FnCmdCopyImageToBuffer = unsafe extern "C" fn(VkCommandBuffer, VkImage, u32, VkBuffer, u32, *const VkBufferImageCopy);
type FnDeviceWaitIdle = unsafe extern "C" fn(VkDevice) -> i32;

struct VkFns {
    destroy_instance: FnDestroyInstance,
    destroy_device: FnDestroyDevice,
    get_device_queue: FnGetDeviceQueue,
    create_buffer: FnCreateBuffer,
    destroy_buffer: FnDestroyBuffer,
    get_buf_mem_reqs: FnGetBufMemReqs,
    alloc_mem: FnAllocMem,
    free_mem: FnFreeMem,
    bind_buf_mem: FnBindBufMem,
    map_mem: FnMapMem,
    unmap_mem: FnUnmapMem,
    create_shader_module: FnCreateShaderModule,
    destroy_shader_module: FnDestroyShaderModule,
    create_desc_set_layout: FnCreateDescSetLayout,
    destroy_desc_set_layout: FnDestroyDescSetLayout,
    create_pipeline_layout: FnCreatePipelineLayout,
    destroy_pipeline_layout: FnDestroyPipelineLayout,
    create_compute_pipelines: FnCreateComputePipelines,
    destroy_pipeline: FnDestroyPipeline,
    create_desc_pool: FnCreateDescPool,
    destroy_desc_pool: FnDestroyDescPool,
    reset_desc_pool: FnResetDescPool,
    alloc_desc_sets: FnAllocDescSets,
    update_desc_sets: FnUpdateDescSets,
    create_cmd_pool: FnCreateCmdPool,
    destroy_cmd_pool: FnDestroyCmdPool,
    alloc_cmd_bufs: FnAllocCmdBufs,
    begin_cmd_buf: FnBeginCmdBuf,
    end_cmd_buf: FnEndCmdBuf,
    cmd_bind_pipeline: FnCmdBindPipeline,
    cmd_bind_desc_sets: FnCmdBindDescSets,
    cmd_dispatch: FnCmdDispatch,
    cmd_pipeline_barrier: FnCmdPipelineBarrier,
    cmd_push_constants: FnCmdPushConstants,
    create_fence: FnCreateFence,
    destroy_fence: FnDestroyFence,
    reset_fences: FnResetFences,
    wait_for_fences: FnWaitForFences,
    queue_submit: FnQueueSubmit,
    // Graphics
    create_image: FnCreateImage,
    destroy_image: FnDestroyImage,
    get_image_mem_reqs: FnGetImageMemReqs,
    bind_image_mem: FnBindImageMem,
    create_image_view: FnCreateImageView,
    destroy_image_view: FnDestroyImageView,
    create_render_pass: FnCreateRenderPass,
    destroy_render_pass: FnDestroyRenderPass,
    create_framebuffer: FnCreateFramebuffer,
    destroy_framebuffer: FnDestroyFramebuffer,
    create_graphics_pipelines: FnCreateGraphicsPipelines,
    cmd_begin_render_pass: FnCmdBeginRenderPass,
    cmd_end_render_pass: FnCmdEndRenderPass,
    cmd_bind_vertex_buffers: FnCmdBindVertexBuffers,
    cmd_draw: FnCmdDraw,
    cmd_set_viewport: FnCmdSetViewport,
    cmd_set_scissor: FnCmdSetScissor,
    cmd_copy_image_to_buffer: FnCmdCopyImageToBuffer,
    device_wait_idle: FnDeviceWaitIdle,
}

// --- GPU buffer ---

pub struct GpuBuf {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    mapped: *mut c_void,
    pub size: usize,
}

// --- GPU vertex for graphics rendering ---

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuVertex {
    pub pos: [f32; 3],
    pub color_packed: u32,
    pub normal: [f32; 3],
}

/// Push constants for GPU lighting (128 bytes, fits Vulkan minimum guarantee)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuPushConstants {
    pub vp: [f32; 16],                    // mat4 VP matrix (64 bytes)
    pub light_dir_ambient: [f32; 4],      // xyz=light_dir, w=ambient
    pub sun_fog_params: [f32; 4],         // x=sun_strength, y=fog_dist_sq_inv, z=fwd_x, w=fwd_z
    pub fog_color: [f32; 4],              // xyz=fog_color (0-1 normalized)
    pub eye_pos: [f32; 4],                // xyz=camera position
}

// --- Render target (offscreen color+depth+readback) ---

struct GpuRenderTarget {
    color_image: VkImage,
    color_memory: VkDeviceMemory,
    color_view: VkImageView,
    depth_image: VkImage,
    depth_memory: VkDeviceMemory,
    depth_view: VkImageView,
    render_pass: VkRenderPass,
    framebuffer: VkFramebufferVk,
    readback_bufs: [GpuBuf; 2], // double-buffered readback
    width: u32,
    height: u32,
}

// --- Graphics pipeline ---

struct GfxPipeline {
    pipeline: VkPipeline,
    layout: VkPipelineLayout,
}

// --- Compute pipeline ---

struct ComputePipeline {
    pipeline: VkPipeline,
    layout: VkPipelineLayout,
    desc_set_layout: VkDescriptorSetLayout,
}

// --- Main GPU context ---

pub struct GpuContext {
    fns: VkFns,
    instance: VkInstance,
    device: VkDevice,
    queue: VkQueue,
    cmd_pool: VkCommandPool,
    cmd_buf: VkCommandBuffer,
    fence: VkFence,
    mem_props: VkPhysicalDeviceMemoryProperties,
    pipelines: Vec<(&'static str, ComputePipeline)>,
    desc_pool: VkDescriptorPool,
    pub device_name: String,
    // Graphics rendering (double-buffered)
    render_target: Option<GpuRenderTarget>,
    gfx_pipeline: Option<GfxPipeline>,
    gfx_cmd_bufs: [VkCommandBuffer; 2],
    gfx_fences: [VkFence; 2],
    gfx_frame_idx: usize,
    gfx_has_prev_frame: bool,
    static_vbuf: Option<GpuBuf>,
    static_vert_count: u32,
    dynamic_vbuf: Option<GpuBuf>,
}

unsafe fn load_fn<T>(get_proc: FnGetInstanceProcAddr, instance: VkInstance, name: &std::ffi::CStr) -> T {
    let p = get_proc(instance, name.as_ptr());
    if p.is_null() { panic!("Vulkan: failed to load {}", name.to_str().unwrap_or("?")); }
    std::mem::transmute_copy(&p)
}

fn find_memory_type(mem_props: &VkPhysicalDeviceMemoryProperties, type_bits: u32, flags: u32) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags & flags == flags {
            return Some(i);
        }
    }
    None
}

impl GpuContext {
    pub fn try_new() -> Option<Self> {
        unsafe { Self::init().ok() }
    }

    /// Initialize GPU with graphics pipeline, exiting if unavailable.
    pub fn init_or_exit(width: u32, height: u32) -> Self {
        let mut ctx = match Self::try_new() {
            Some(c) => {
                eprintln!("GPU: {}", c.device_name);
                c
            }
            None => {
                eprintln!("Error: No Vulkan GPU available.");
                std::process::exit(1);
            }
        };
        ctx.init_graphics(width, height);
        if !ctx.has_graphics() {
            eprintln!("Error: GPU graphics pipeline failed to initialize.");
            std::process::exit(1);
        }
        eprintln!("GPU graphics pipeline ready ({}x{})", width, height);
        ctx
    }

    unsafe fn init() -> Result<Self, &'static str> {
        let lib = dlopen(c"libvulkan.so.1".as_ptr(), RTLD_LAZY);
        if lib.is_null() { return Err("no libvulkan.so.1"); }

        let get_proc: FnGetInstanceProcAddr = {
            let p = dlsym(lib, c"vkGetInstanceProcAddr".as_ptr());
            if p.is_null() { return Err("no vkGetInstanceProcAddr"); }
            std::mem::transmute(p)
        };

        // Create instance
        let create_instance: FnCreateInstance = load_fn(get_proc, ptr::null_mut(), c"vkCreateInstance");

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            app_name: c"Clauding".as_ptr(),
            app_version: 1,
            engine_name: c"Clauding".as_ptr(),
            engine_version: 1,
            api_version: VK_API_VERSION_1_1,
        };

        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            app_info: &app_info,
            layer_count: 0,
            layers: ptr::null(),
            ext_count: 0,
            exts: ptr::null(),
        };

        let mut instance: VkInstance = ptr::null_mut();
        let res = create_instance(&create_info, ptr::null(), &mut instance);
        if res != VK_SUCCESS { return Err("vkCreateInstance failed"); }

        // Load instance-level functions
        let enum_phys_devs: FnEnumPhysDevices = load_fn(get_proc, instance, c"vkEnumeratePhysicalDevices");
        let get_phys_props: FnGetPhysDevProps = load_fn(get_proc, instance, c"vkGetPhysicalDeviceProperties");
        let get_phys_mem: FnGetPhysDevMemProps = load_fn(get_proc, instance, c"vkGetPhysicalDeviceMemoryProperties");
        let get_phys_queue: FnGetPhysDevQueueFamProps = load_fn(get_proc, instance, c"vkGetPhysicalDeviceQueueFamilyProperties");
        let create_device: FnCreateDevice = load_fn(get_proc, instance, c"vkCreateDevice");

        // Pick physical device (prefer discrete GPU)
        let mut count = 0u32;
        enum_phys_devs(instance, &mut count, ptr::null_mut());
        if count == 0 { return Err("no Vulkan physical devices"); }
        let mut devs = vec![ptr::null_mut(); count as usize];
        enum_phys_devs(instance, &mut count, devs.as_mut_ptr());

        let mut chosen: VkPhysicalDevice = ptr::null_mut();
        let mut device_name = String::from("unknown");
        let mut chosen_queue_family = u32::MAX;

        for &dev in &devs {
            let mut props = [0u8; 1024];
            get_phys_props(dev, &mut props);
            let dev_type = u32::from_ne_bytes(props[16..20].try_into().unwrap());
            let name_end = props[20..276].iter().position(|&b| b == 0).unwrap_or(255);
            let name = std::str::from_utf8(&props[20..20+name_end]).unwrap_or("?").to_string();

            // Find compute queue family
            let mut qf_count = 0u32;
            get_phys_queue(dev, &mut qf_count, ptr::null_mut());
            let mut qf_props = vec![VkQueueFamilyProperties {
                queue_flags: 0, queue_count: 0, timestamp_valid_bits: 0,
                min_image_transfer_granularity: [0; 3],
            }; qf_count as usize];
            get_phys_queue(dev, &mut qf_count, qf_props.as_mut_ptr());

            for (i, qf) in qf_props.iter().enumerate() {
                if qf.queue_flags & VK_QUEUE_GRAPHICS_BIT != 0 {
                    if chosen.is_null() || dev_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
                        chosen = dev;
                        device_name = name.clone();
                        chosen_queue_family = i as u32;
                    }
                    break;
                }
            }
        }

        if chosen.is_null() { return Err("no compute-capable GPU"); }

        // Get memory properties
        let mut mem_props = std::mem::zeroed::<VkPhysicalDeviceMemoryProperties>();
        get_phys_mem(chosen, &mut mem_props);

        // Create logical device
        let priority = 1.0f32;
        let queue_info = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index: chosen_queue_family,
            queue_count: 1,
            p_queue_priorities: &priority,
        };

        let dev_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            layer_count: 0,
            layers: ptr::null(),
            ext_count: 0,
            exts: ptr::null(),
            features: ptr::null(),
        };

        let mut device: VkDevice = ptr::null_mut();
        let res = create_device(chosen, &dev_info, ptr::null(), &mut device);
        if res != VK_SUCCESS { return Err("vkCreateDevice failed"); }

        // Load device-level functions
        let fns = VkFns {
            destroy_instance: load_fn(get_proc, instance, c"vkDestroyInstance"),
            destroy_device: load_fn(get_proc, instance, c"vkDestroyDevice"),
            get_device_queue: load_fn(get_proc, instance, c"vkGetDeviceQueue"),
            create_buffer: load_fn(get_proc, instance, c"vkCreateBuffer"),
            destroy_buffer: load_fn(get_proc, instance, c"vkDestroyBuffer"),
            get_buf_mem_reqs: load_fn(get_proc, instance, c"vkGetBufferMemoryRequirements"),
            alloc_mem: load_fn(get_proc, instance, c"vkAllocateMemory"),
            free_mem: load_fn(get_proc, instance, c"vkFreeMemory"),
            bind_buf_mem: load_fn(get_proc, instance, c"vkBindBufferMemory"),
            map_mem: load_fn(get_proc, instance, c"vkMapMemory"),
            unmap_mem: load_fn(get_proc, instance, c"vkUnmapMemory"),
            create_shader_module: load_fn(get_proc, instance, c"vkCreateShaderModule"),
            destroy_shader_module: load_fn(get_proc, instance, c"vkDestroyShaderModule"),
            create_desc_set_layout: load_fn(get_proc, instance, c"vkCreateDescriptorSetLayout"),
            destroy_desc_set_layout: load_fn(get_proc, instance, c"vkDestroyDescriptorSetLayout"),
            create_pipeline_layout: load_fn(get_proc, instance, c"vkCreatePipelineLayout"),
            destroy_pipeline_layout: load_fn(get_proc, instance, c"vkDestroyPipelineLayout"),
            create_compute_pipelines: load_fn(get_proc, instance, c"vkCreateComputePipelines"),
            destroy_pipeline: load_fn(get_proc, instance, c"vkDestroyPipeline"),
            create_desc_pool: load_fn(get_proc, instance, c"vkCreateDescriptorPool"),
            destroy_desc_pool: load_fn(get_proc, instance, c"vkDestroyDescriptorPool"),
            reset_desc_pool: load_fn(get_proc, instance, c"vkResetDescriptorPool"),
            alloc_desc_sets: load_fn(get_proc, instance, c"vkAllocateDescriptorSets"),
            update_desc_sets: load_fn(get_proc, instance, c"vkUpdateDescriptorSets"),
            create_cmd_pool: load_fn(get_proc, instance, c"vkCreateCommandPool"),
            destroy_cmd_pool: load_fn(get_proc, instance, c"vkDestroyCommandPool"),
            alloc_cmd_bufs: load_fn(get_proc, instance, c"vkAllocateCommandBuffers"),
            begin_cmd_buf: load_fn(get_proc, instance, c"vkBeginCommandBuffer"),
            end_cmd_buf: load_fn(get_proc, instance, c"vkEndCommandBuffer"),
            cmd_bind_pipeline: load_fn(get_proc, instance, c"vkCmdBindPipeline"),
            cmd_bind_desc_sets: load_fn(get_proc, instance, c"vkCmdBindDescriptorSets"),
            cmd_dispatch: load_fn(get_proc, instance, c"vkCmdDispatch"),
            cmd_pipeline_barrier: load_fn(get_proc, instance, c"vkCmdPipelineBarrier"),
            cmd_push_constants: load_fn(get_proc, instance, c"vkCmdPushConstants"),
            create_fence: load_fn(get_proc, instance, c"vkCreateFence"),
            destroy_fence: load_fn(get_proc, instance, c"vkDestroyFence"),
            reset_fences: load_fn(get_proc, instance, c"vkResetFences"),
            wait_for_fences: load_fn(get_proc, instance, c"vkWaitForFences"),
            queue_submit: load_fn(get_proc, instance, c"vkQueueSubmit"),
            // Graphics
            create_image: load_fn(get_proc, instance, c"vkCreateImage"),
            destroy_image: load_fn(get_proc, instance, c"vkDestroyImage"),
            get_image_mem_reqs: load_fn(get_proc, instance, c"vkGetImageMemoryRequirements"),
            bind_image_mem: load_fn(get_proc, instance, c"vkBindImageMemory"),
            create_image_view: load_fn(get_proc, instance, c"vkCreateImageView"),
            destroy_image_view: load_fn(get_proc, instance, c"vkDestroyImageView"),
            create_render_pass: load_fn(get_proc, instance, c"vkCreateRenderPass"),
            destroy_render_pass: load_fn(get_proc, instance, c"vkDestroyRenderPass"),
            create_framebuffer: load_fn(get_proc, instance, c"vkCreateFramebuffer"),
            destroy_framebuffer: load_fn(get_proc, instance, c"vkDestroyFramebuffer"),
            create_graphics_pipelines: load_fn(get_proc, instance, c"vkCreateGraphicsPipelines"),
            cmd_begin_render_pass: load_fn(get_proc, instance, c"vkCmdBeginRenderPass"),
            cmd_end_render_pass: load_fn(get_proc, instance, c"vkCmdEndRenderPass"),
            cmd_bind_vertex_buffers: load_fn(get_proc, instance, c"vkCmdBindVertexBuffers"),
            cmd_draw: load_fn(get_proc, instance, c"vkCmdDraw"),
            cmd_set_viewport: load_fn(get_proc, instance, c"vkCmdSetViewport"),
            cmd_set_scissor: load_fn(get_proc, instance, c"vkCmdSetScissor"),
            cmd_copy_image_to_buffer: load_fn(get_proc, instance, c"vkCmdCopyImageToBuffer"),
            device_wait_idle: load_fn(get_proc, instance, c"vkDeviceWaitIdle"),
        };

        // Get queue
        let mut queue: VkQueue = ptr::null_mut();
        (fns.get_device_queue)(device, chosen_queue_family, 0, &mut queue);

        // Create command pool
        let pool_info = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: chosen_queue_family,
        };
        let mut cmd_pool: VkCommandPool = 0;
        let res = (fns.create_cmd_pool)(device, &pool_info, ptr::null(), &mut cmd_pool);
        if res != VK_SUCCESS { return Err("vkCreateCommandPool failed"); }

        // Allocate command buffer
        let alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: cmd_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd_buf: VkCommandBuffer = ptr::null_mut();
        (fns.alloc_cmd_bufs)(device, &alloc_info, &mut cmd_buf);

        // Create fence (for compute)
        let fence_info = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
        };
        let mut fence: VkFence = 0;
        (fns.create_fence)(device, &fence_info, ptr::null(), &mut fence);

        // Graphics double-buffered command buffers + fences
        let gfx_alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: cmd_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 2,
        };
        let mut gfx_cmd_bufs: [VkCommandBuffer; 2] = [ptr::null_mut(); 2];
        (fns.alloc_cmd_bufs)(device, &gfx_alloc_info, gfx_cmd_bufs.as_mut_ptr());

        let fence_info_signaled = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: VK_FENCE_CREATE_SIGNALED_BIT,
        };
        let mut gfx_fences: [VkFence; 2] = [0; 2];
        (fns.create_fence)(device, &fence_info_signaled, ptr::null(), &mut gfx_fences[0]);
        (fns.create_fence)(device, &fence_info_signaled, ptr::null(), &mut gfx_fences[1]);

        // Create descriptor pool (enough for all our pipelines)
        let pool_size = VkDescriptorPoolSize {
            typ: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 64,
        };
        let desc_pool_info = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets: 16,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut desc_pool: VkDescriptorPool = 0;
        (fns.create_desc_pool)(device, &desc_pool_info, ptr::null(), &mut desc_pool);

        let mut ctx = GpuContext {
            fns,
            instance,
            device,
            queue,
            cmd_pool,
            cmd_buf,
            fence,
            mem_props,
            pipelines: Vec::new(),
            desc_pool,
            device_name,
            render_target: None,
            gfx_pipeline: None,
            gfx_cmd_bufs,
            gfx_fences,
            gfx_frame_idx: 0,
            gfx_has_prev_frame: false,
            static_vbuf: None,
            static_vert_count: 0,
            dynamic_vbuf: None,
        };

        // Build pipelines
        ctx.add_pipeline("test_multiply", &gpu_kernels::build_test_multiply(), 1, 4)?;
        ctx.add_pipeline("particle_update", &gpu_kernels::build_particle_update(), 7, 12)?;

        Ok(ctx)
    }

    unsafe fn add_pipeline(&mut self, name: &'static str, spirv: &[u32], binding_count: u32, push_size: u32) -> Result<(), &'static str> {
        // Create shader module
        let module = self.create_shader_module(spirv);

        // Descriptor set layout
        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..binding_count).map(|i| VkDescriptorSetLayoutBinding {
            binding: i,
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            p_immutable_samplers: ptr::null(),
        }).collect();

        let layout_info = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count,
            p_bindings: bindings.as_ptr(),
        };
        let mut desc_layout: VkDescriptorSetLayout = 0;
        (self.fns.create_desc_set_layout)(self.device, &layout_info, ptr::null(), &mut desc_layout);

        // Pipeline layout
        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_size,
        };
        let pipe_layout_info = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &desc_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };
        let mut pipe_layout: VkPipelineLayout = 0;
        (self.fns.create_pipeline_layout)(self.device, &pipe_layout_info, ptr::null(), &mut pipe_layout);

        // Compute pipeline
        let stage_info = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module,
            p_name: c"main".as_ptr(),
            p_specialization_info: ptr::null(),
        };
        let pipeline_info = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: stage_info,
            layout: pipe_layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut pipeline: VkPipeline = 0;
        let res = (self.fns.create_compute_pipelines)(self.device, 0, 1, &pipeline_info, ptr::null(), &mut pipeline);
        if res != VK_SUCCESS { return Err("vkCreateComputePipelines failed"); }

        (self.fns.destroy_shader_module)(self.device, module, ptr::null());

        self.pipelines.push((name, ComputePipeline {
            pipeline,
            layout: pipe_layout,
            desc_set_layout: desc_layout,
        }));

        Ok(())
    }

    pub fn create_buffer(&self, size_bytes: usize) -> GpuBuf {
        let flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        self.create_buffer_usage_prefer(size_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, flags, flags)
    }

    pub fn upload(&self, buf: &GpuBuf, data: &[u8]) {
        unsafe {
            let len = data.len().min(buf.size);
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.mapped as *mut u8, len);
        }
    }

    pub fn download(&self, buf: &GpuBuf, data: &mut [u8]) {
        unsafe {
            let len = data.len().min(buf.size);
            std::ptr::copy_nonoverlapping(buf.mapped as *const u8, data.as_mut_ptr(), len);
        }
    }

    pub fn dispatch(&mut self, kernel: &str, buffers: &[&GpuBuf], push_constants: &[u8], count: u32) {
        unsafe {
            // Reset descriptor pool (previous dispatch is done since we waited on fence)
            (self.fns.reset_desc_pool)(self.device, self.desc_pool, 0);

            let pipe = &self.pipelines.iter().find(|(n, _)| *n == kernel)
                .expect("Unknown kernel").1;

            // Allocate descriptor set
            let alloc_info = VkDescriptorSetAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: &pipe.desc_set_layout,
            };
            let mut desc_set: VkDescriptorSet = 0;
            (self.fns.alloc_desc_sets)(self.device, &alloc_info, &mut desc_set);

            // Update descriptor set with buffer bindings
            let buf_infos: Vec<VkDescriptorBufferInfo> = buffers.iter().map(|b| VkDescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: VK_WHOLE_SIZE,
            }).collect();

            let writes: Vec<VkWriteDescriptorSet> = buf_infos.iter().enumerate().map(|(i, info)| VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: desc_set,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: info,
                p_texel_buffer_view: ptr::null(),
            }).collect();

            (self.fns.update_desc_sets)(self.device, writes.len() as u32, writes.as_ptr(), 0, ptr::null());

            // Record command buffer
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            (self.fns.begin_cmd_buf)(self.cmd_buf, &begin_info);
            (self.fns.cmd_bind_pipeline)(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
            (self.fns.cmd_bind_desc_sets)(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipe.layout, 0, 1, &desc_set, 0, ptr::null());

            if !push_constants.is_empty() {
                (self.fns.cmd_push_constants)(self.cmd_buf, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, push_constants.len() as u32, push_constants.as_ptr() as *const c_void);
            }

            let groups = (count + 63) / 64;
            (self.fns.cmd_dispatch)(self.cmd_buf, groups, 1, 1);

            // Memory barrier
            let barriers: Vec<VkBufferMemoryBarrier> = buffers.iter().map(|b| VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: b.buffer,
                offset: 0,
                size: VK_WHOLE_SIZE,
            }).collect();
            (self.fns.cmd_pipeline_barrier)(self.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, ptr::null(), barriers.len() as u32, barriers.as_ptr(), 0, ptr::null());

            (self.fns.end_cmd_buf)(self.cmd_buf);

            // Submit
            let submit = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &self.cmd_buf,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            (self.fns.reset_fences)(self.device, 1, &self.fence);
            (self.fns.queue_submit)(self.queue, 1, &submit, self.fence);
            (self.fns.wait_for_fences)(self.device, 1, &self.fence, 1, u64::MAX);
        }
    }

    pub fn free_buffer(&self, buf: GpuBuf) {
        unsafe {
            (self.fns.unmap_mem)(self.device, buf.memory);
            (self.fns.destroy_buffer)(self.device, buf.buffer, ptr::null());
            (self.fns.free_mem)(self.device, buf.memory, ptr::null());
        }
    }

    /// Allocate device memory of the given size and type
    unsafe fn alloc_device_memory(&self, size: u64, mem_type_index: u32) -> VkDeviceMemory {
        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: size,
            memory_type_index: mem_type_index,
        };
        let mut memory: VkDeviceMemory = 0;
        (self.fns.alloc_mem)(self.device, &alloc_info, ptr::null(), &mut memory);
        memory
    }

    /// Create buffer, trying preferred memory flags first, then falling back
    fn create_buffer_usage_prefer(&self, size_bytes: usize, usage: u32, preferred: u32, fallback: u32) -> GpuBuf {
        unsafe {
            let buf_info = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                size: size_bytes as u64,
                usage,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
            };
            let mut buffer: VkBuffer = 0;
            (self.fns.create_buffer)(self.device, &buf_info, ptr::null(), &mut buffer);

            let mut reqs = std::mem::zeroed::<VkMemoryRequirements>();
            (self.fns.get_buf_mem_reqs)(self.device, buffer, &mut reqs);

            let mem_type = find_memory_type(&self.mem_props, reqs.memory_type_bits, preferred)
                .or_else(|| find_memory_type(&self.mem_props, reqs.memory_type_bits, fallback))
                .expect("No suitable memory type");

            let memory = self.alloc_device_memory(reqs.size, mem_type);
            (self.fns.bind_buf_mem)(self.device, buffer, memory, 0);

            let mut mapped: *mut c_void = ptr::null_mut();
            if preferred & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 || fallback & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 {
                (self.fns.map_mem)(self.device, memory, 0, VK_WHOLE_SIZE, 0, &mut mapped);
            }

            GpuBuf { buffer, memory, mapped, size: size_bytes }
        }
    }

    unsafe fn create_image_alloc(&self, width: u32, height: u32, format: u32, usage: u32) -> (VkImage, VkDeviceMemory) {
        let img_info = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format,
            extent: VkExtent3D { width, height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        let mut image: VkImage = 0;
        let res = (self.fns.create_image)(self.device, &img_info, ptr::null(), &mut image);
        if res != VK_SUCCESS { panic!("vkCreateImage failed: {}", res); }

        let mut reqs = std::mem::zeroed::<VkMemoryRequirements>();
        (self.fns.get_image_mem_reqs)(self.device, image, &mut reqs);

        let mem_type = find_memory_type(&self.mem_props, reqs.memory_type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            .expect("No device-local memory for image");

        let memory = self.alloc_device_memory(reqs.size, mem_type);
        (self.fns.bind_image_mem)(self.device, image, memory, 0);

        (image, memory)
    }

    unsafe fn create_image_view(&self, image: VkImage, format: u32, aspect: u32) -> VkImageView {
        let view_info = VkImageViewCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            image,
            view_type: VK_IMAGE_VIEW_TYPE_2D,
            format,
            components: VkComponentMapping {
                r: VK_COMPONENT_SWIZZLE_IDENTITY,
                g: VK_COMPONENT_SWIZZLE_IDENTITY,
                b: VK_COMPONENT_SWIZZLE_IDENTITY,
                a: VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            subresource_range: VkImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        let mut view: VkImageView = 0;
        (self.fns.create_image_view)(self.device, &view_info, ptr::null(), &mut view);
        view
    }

    /// Initialize graphics rendering pipeline for given resolution
    pub fn init_graphics(&mut self, width: u32, height: u32) {
        unsafe {
            // Create render pass
            let attachments = [
                // Color attachment
                VkAttachmentDescription {
                    flags: 0,
                    format: VK_FORMAT_B8G8R8A8_UNORM,
                    samples: VK_SAMPLE_COUNT_1_BIT,
                    load_op: VK_ATTACHMENT_LOAD_OP_CLEAR,
                    store_op: VK_ATTACHMENT_STORE_OP_STORE,
                    stencil_load_op: VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencil_store_op: VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
                    final_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                },
                // Depth attachment
                VkAttachmentDescription {
                    flags: 0,
                    format: VK_FORMAT_D32_SFLOAT,
                    samples: VK_SAMPLE_COUNT_1_BIT,
                    load_op: VK_ATTACHMENT_LOAD_OP_CLEAR,
                    store_op: VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    stencil_load_op: VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencil_store_op: VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
                    final_layout: VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
            ];

            let color_ref = VkAttachmentReference {
                attachment: 0,
                layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            let depth_ref = VkAttachmentReference {
                attachment: 1,
                layout: VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpass = VkSubpassDescription {
                flags: 0,
                pipeline_bind_point: VK_PIPELINE_BIND_POINT_GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                color_attachment_count: 1,
                p_color_attachments: &color_ref,
                p_resolve_attachments: ptr::null(),
                p_depth_stencil_attachment: &depth_ref,
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            };

            let dependency = VkSubpassDependency {
                src_subpass: VK_SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                dst_stage_mask: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                dependency_flags: 0,
            };

            let rp_info = VkRenderPassCreateInfo {
                s_type: VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                attachment_count: 2,
                p_attachments: attachments.as_ptr(),
                subpass_count: 1,
                p_subpasses: &subpass,
                dependency_count: 1,
                p_dependencies: &dependency,
            };

            let mut render_pass: VkRenderPass = 0;
            let res = (self.fns.create_render_pass)(self.device, &rp_info, ptr::null(), &mut render_pass);
            if res != VK_SUCCESS { panic!("vkCreateRenderPass failed: {}", res); }

            // Create graphics pipeline
            let vert_spirv = gpu_shaders::build_vertex_shader();
            let frag_spirv = gpu_shaders::build_fragment_shader();

            let vert_module = self.create_shader_module(&vert_spirv);
            let frag_module = self.create_shader_module(&frag_spirv);

            let stages = [
                VkPipelineShaderStageCreateInfo {
                    s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_VERTEX_BIT,
                    module: vert_module,
                    p_name: c"main".as_ptr(),
                    p_specialization_info: ptr::null(),
                },
                VkPipelineShaderStageCreateInfo {
                    s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_FRAGMENT_BIT,
                    module: frag_module,
                    p_name: c"main".as_ptr(),
                    p_specialization_info: ptr::null(),
                },
            ];

            // Vertex input: GpuVertex { pos: [f32;3], color_packed: u32, normal: [f32;3] }
            let binding = VkVertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<GpuVertex>() as u32,
                input_rate: 0, // VK_VERTEX_INPUT_RATE_VERTEX
            };
            let attributes = [
                VkVertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: VK_FORMAT_R32G32B32_SFLOAT, // pos
                    offset: 0,
                },
                VkVertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: VK_FORMAT_B8G8R8A8_UNORM, // color_packed (auto-unpacked to vec4)
                    offset: 12,
                },
                VkVertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: VK_FORMAT_R32G32B32_SFLOAT, // normal
                    offset: 16,
                },
            ];

            let vertex_input = VkPipelineVertexInputStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &binding,
                vertex_attribute_description_count: 3,
                p_vertex_attribute_descriptions: attributes.as_ptr(),
            };

            let input_assembly = VkPipelineInputAssemblyStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                primitive_restart_enable: 0,
            };

            let viewport_state = VkPipelineViewportStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                viewport_count: 1,
                p_viewports: ptr::null(), // dynamic
                scissor_count: 1,
                p_scissors: ptr::null(), // dynamic
            };

            let rasterization = VkPipelineRasterizationStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                depth_clamp_enable: 0,
                rasterizer_discard_enable: 0,
                polygon_mode: VK_POLYGON_MODE_FILL,
                cull_mode: VK_CULL_MODE_NONE,
                front_face: _VK_FRONT_FACE_COUNTER_CLOCKWISE, // CCW mesh winding stays CCW in framebuffer (proj Y-flip + Vulkan Y-down cancel)
                depth_bias_enable: 0,
                depth_bias_constant_factor: 0.0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_factor: 0.0,
                line_width: 1.0,
            };

            let multisample = VkPipelineMultisampleStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                rasterization_samples: VK_SAMPLE_COUNT_1_BIT,
                sample_shading_enable: 0,
                min_sample_shading: 1.0,
                p_sample_mask: ptr::null(),
                alpha_to_coverage_enable: 0,
                alpha_to_one_enable: 0,
            };

            let stencil_nop = VkStencilOpState {
                fail_op: 0, pass_op: 0, depth_fail_op: 0, compare_op: 0,
                compare_mask: 0, write_mask: 0, reference: 0,
            };
            let depth_stencil = VkPipelineDepthStencilStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: VK_COMPARE_OP_LESS,
                depth_bounds_test_enable: 0,
                stencil_test_enable: 0,
                front: stencil_nop,
                back: stencil_nop,
                min_depth_bounds: 0.0,
                max_depth_bounds: 1.0,
            };

            let blend_attachment = VkPipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: 0,
                dst_color_blend_factor: 0,
                color_blend_op: 0,
                src_alpha_blend_factor: 0,
                dst_alpha_blend_factor: 0,
                alpha_blend_op: 0,
                color_write_mask: 0xF, // RGBA
            };
            let color_blend = VkPipelineColorBlendStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                logic_op_enable: 0,
                logic_op: VK_LOGIC_OP_COPY,
                attachment_count: 1,
                p_attachments: &blend_attachment,
                blend_constants: [0.0; 4],
            };

            let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
            let dynamic_state = VkPipelineDynamicStateCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                dynamic_state_count: 2,
                p_dynamic_states: dynamic_states.as_ptr(),
            };

            // Pipeline layout: push constants = 128 bytes (VP + lighting) at vertex stage
            let push_range = VkPushConstantRange {
                stage_flags: VK_SHADER_STAGE_VERTEX_BIT,
                offset: 0,
                size: 128,
            };
            let pipe_layout_info = VkPipelineLayoutCreateInfo {
                s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                set_layout_count: 0,
                p_set_layouts: ptr::null(),
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_range,
            };
            let mut gfx_layout: VkPipelineLayout = 0;
            (self.fns.create_pipeline_layout)(self.device, &pipe_layout_info, ptr::null(), &mut gfx_layout);

            let gfx_pipeline_info = VkGraphicsPipelineCreateInfo {
                s_type: VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                stage_count: 2,
                p_stages: stages.as_ptr(),
                p_vertex_input_state: &vertex_input,
                p_input_assembly_state: &input_assembly,
                p_tessellation_state: ptr::null(),
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterization,
                p_multisample_state: &multisample,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &color_blend,
                p_dynamic_state: &dynamic_state,
                layout: gfx_layout,
                render_pass,
                subpass: 0,
                base_pipeline_handle: 0,
                base_pipeline_index: -1,
            };

            let mut gfx_pipeline: VkPipeline = 0;
            let res = (self.fns.create_graphics_pipelines)(self.device, 0, 1, &gfx_pipeline_info, ptr::null(), &mut gfx_pipeline);
            if res != VK_SUCCESS { panic!("vkCreateGraphicsPipelines failed: {}", res); }

            (self.fns.destroy_shader_module)(self.device, vert_module, ptr::null());
            (self.fns.destroy_shader_module)(self.device, frag_module, ptr::null());

            self.gfx_pipeline = Some(GfxPipeline { pipeline: gfx_pipeline, layout: gfx_layout });

            // Create render target at initial resolution
            self.create_render_target(width, height, render_pass);

            // Create initial vertex buffer (4MB, grows as needed)
            let vbuf_size = 4 * 1024 * 1024;
            self.dynamic_vbuf = Some(self.create_buffer_usage_prefer(
                vbuf_size,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ));
        }
    }

    unsafe fn create_shader_module(&self, spirv: &[u32]) -> VkShaderModule {
        let info = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
        };
        let mut module: VkShaderModule = 0;
        let res = (self.fns.create_shader_module)(self.device, &info, ptr::null(), &mut module);
        if res != VK_SUCCESS { panic!("vkCreateShaderModule failed: {}", res); }
        module
    }

    unsafe fn create_render_target(&mut self, width: u32, height: u32, render_pass: VkRenderPass) {
        // Color image
        let (color_image, color_memory) = self.create_image_alloc(
            width, height, VK_FORMAT_B8G8R8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        );
        let color_view = self.create_image_view(color_image, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

        // Depth image
        let (depth_image, depth_memory) = self.create_image_alloc(
            width, height, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        );
        let depth_view = self.create_image_view(depth_image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

        // Framebuffer
        let attachments = [color_view, depth_view];
        let fb_info = VkFramebufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            render_pass,
            attachment_count: 2,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
        };
        let mut framebuffer: VkFramebufferVk = 0;
        (self.fns.create_framebuffer)(self.device, &fb_info, ptr::null(), &mut framebuffer);

        // Double-buffered readback — prefer HOST_CACHED for fast CPU reads
        let readback_size = (width * height * 4) as usize;
        let readback_flags_pref = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        let readback_flags_fall = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let readback_buf_0 = self.create_buffer_usage_prefer(readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_flags_pref, readback_flags_fall);
        let readback_buf_1 = self.create_buffer_usage_prefer(readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_flags_pref, readback_flags_fall);

        self.render_target = Some(GpuRenderTarget {
            color_image,
            color_memory,
            color_view,
            depth_image,
            depth_memory,
            depth_view,
            render_pass,
            framebuffer,
            readback_bufs: [readback_buf_0, readback_buf_1],
            width,
            height,
        });
    }

    fn destroy_render_target_resources(&mut self) {
        unsafe {
            if let Some(rt) = self.render_target.take() {
                (self.fns.destroy_framebuffer)(self.device, rt.framebuffer, ptr::null());
                (self.fns.destroy_image_view)(self.device, rt.color_view, ptr::null());
                (self.fns.destroy_image_view)(self.device, rt.depth_view, ptr::null());
                (self.fns.destroy_image)(self.device, rt.color_image, ptr::null());
                (self.fns.free_mem)(self.device, rt.color_memory, ptr::null());
                (self.fns.destroy_image)(self.device, rt.depth_image, ptr::null());
                (self.fns.free_mem)(self.device, rt.depth_memory, ptr::null());
                for rb in &rt.readback_bufs {
                    (self.fns.unmap_mem)(self.device, rb.memory);
                    (self.fns.destroy_buffer)(self.device, rb.buffer, ptr::null());
                    (self.fns.free_mem)(self.device, rb.memory, ptr::null());
                }
            }
        }
    }

    /// Resize render target when window changes
    pub fn resize_render_target(&mut self, width: u32, height: u32) {
        let render_pass = match &self.render_target {
            Some(rt) => rt.render_pass,
            None => return,
        };
        // Don't recreate if same size
        if let Some(ref rt) = self.render_target {
            if rt.width == width && rt.height == height { return; }
        }
        self.destroy_render_target_resources();
        unsafe {
            self.create_render_target(width, height, render_pass);
        }
    }

    /// Upload static vertex data (call once or when lighting changes)
    pub fn upload_static_vertices(&mut self, vertices: &[GpuVertex]) {
        let vert_bytes = vertices.len() * std::mem::size_of::<GpuVertex>();
        if vert_bytes == 0 {
            self.static_vert_count = 0;
            return;
        }

        // Wait for in-flight GPU work that may reference the old buffer
        unsafe {
            (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[0], 1, u64::MAX);
            (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[1], 1, u64::MAX);
        }

        // Ensure static buffer is large enough
        let need_realloc = match &self.static_vbuf {
            Some(vbuf) => vbuf.size < vert_bytes,
            None => true,
        };
        if need_realloc {
            if let Some(old) = self.static_vbuf.take() {
                self.free_buffer(old);
            }
            // Prefer DEVICE_LOCAL + HOST_VISIBLE (resizable BAR → GPU VRAM, CPU-mappable)
            // Fallback: HOST_VISIBLE + HOST_COHERENT (system RAM, PCIe reads every frame)
            self.static_vbuf = Some(self.create_buffer_usage_prefer(
                vert_bytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ));
        }
        let vbuf = self.static_vbuf.as_ref().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                vbuf.mapped as *mut u8,
                vert_bytes,
            );
        }
        self.static_vert_count = vertices.len() as u32;
    }

    /// GPU-render static + dynamic vertices to output pixel buffer (double-buffered)
    ///
    /// Submits current frame's GPU work and reads back the previous frame's result.
    /// This overlaps GPU rendering with CPU work, hiding the fence wait.
    /// Output contains the PREVIOUS frame's pixels (1 frame latency).
    pub fn render_frame(
        &mut self,
        dynamic_verts: &[GpuVertex],
        push_constants: &GpuPushConstants,
        clear_color: [f32; 4],
        width: u32,
        height: u32,
        output: &mut [u32],
    ) {
        if self.render_target.is_none() || self.gfx_pipeline.is_none() { return; }

        let has_static = self.static_vbuf.is_some() && self.static_vert_count > 0;
        let has_dynamic = !dynamic_verts.is_empty();
        if !has_static && !has_dynamic { return; }

        // Resize render target if needed
        {
            let rt = self.render_target.as_ref().unwrap();
            if rt.width != width || rt.height != height {
                let rp = rt.render_pass;
                // Wait for any in-flight work before destroying resources
                unsafe {
                    (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[0], 1, u64::MAX);
                    (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[1], 1, u64::MAX);
                }
                self.gfx_has_prev_frame = false;
                self.destroy_render_target_resources();
                unsafe { self.create_render_target(width, height, rp); }
            }
        }

        let curr = self.gfx_frame_idx;
        let prev = 1 - curr;

        unsafe {
            // Wait for current slot's fence (ensures this cmd_buf is free to record)
            // On first frame, fences are pre-signaled so this returns immediately
            (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[curr], 1, u64::MAX);

            // Read back PREVIOUS frame's result (if we have one)
            if self.gfx_has_prev_frame {
                let rt = self.render_target.as_ref().unwrap();
                let pixel_count = (width * height) as usize;
                let copy_bytes = pixel_count.min(output.len()) * 4;
                std::ptr::copy_nonoverlapping(
                    rt.readback_bufs[prev].mapped as *const u8,
                    output.as_mut_ptr() as *mut u8,
                    copy_bytes,
                );
            }

            // Upload dynamic vertices
            let dyn_vert_count;
            if has_dynamic {
                let vert_bytes = dynamic_verts.len() * std::mem::size_of::<GpuVertex>();
                if let Some(ref vbuf) = self.dynamic_vbuf {
                    if vbuf.size < vert_bytes {
                        let old = self.dynamic_vbuf.take().unwrap();
                        self.free_buffer(old);
                        self.dynamic_vbuf = Some(self.create_buffer_usage_prefer(
                            vert_bytes * 2,
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        ));
                    }
                }
                let vbuf = self.dynamic_vbuf.as_ref().unwrap();
                std::ptr::copy_nonoverlapping(
                    dynamic_verts.as_ptr() as *const u8,
                    vbuf.mapped as *mut u8,
                    vert_bytes,
                );
                dyn_vert_count = dynamic_verts.len() as u32;
            } else {
                dyn_vert_count = 0;
            }

            let rt = self.render_target.as_ref().unwrap();
            let gfx = self.gfx_pipeline.as_ref().unwrap();
            let gfx_pipeline = gfx.pipeline;
            let gfx_layout = gfx.layout;
            let cmd = self.gfx_cmd_bufs[curr];

            // Record command buffer
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            (self.fns.begin_cmd_buf)(cmd, &begin_info);

            // Begin render pass
            let clear_values = [
                VkClearValue { color: VkClearColorValue { float32: clear_color } },
                VkClearValue { depth_stencil: VkClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];
            let rp_begin = VkRenderPassBeginInfo {
                s_type: VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass: rt.render_pass,
                framebuffer: rt.framebuffer,
                render_area: VkRect2D { offset_x: 0, offset_y: 0, extent_w: width, extent_h: height },
                clear_value_count: 2,
                p_clear_values: clear_values.as_ptr(),
            };
            (self.fns.cmd_begin_render_pass)(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

            // Bind pipeline + dynamic state
            (self.fns.cmd_bind_pipeline)(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_pipeline);
            let viewport = VkViewport {
                x: 0.0, y: 0.0,
                width: width as f32, height: height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            (self.fns.cmd_set_viewport)(cmd, 0, 1, &viewport);
            let scissor = VkRect2D { offset_x: 0, offset_y: 0, extent_w: width, extent_h: height };
            (self.fns.cmd_set_scissor)(cmd, 0, 1, &scissor);
            (self.fns.cmd_push_constants)(
                cmd, gfx_layout, VK_SHADER_STAGE_VERTEX_BIT,
                0, 128, push_constants as *const GpuPushConstants as *const c_void,
            );

            // Draw static vertices (single draw call — GPU handles 1.15M verts efficiently)
            if has_static {
                let svbuf = self.static_vbuf.as_ref().unwrap();
                let offset: VkDeviceSize = 0;
                (self.fns.cmd_bind_vertex_buffers)(cmd, 0, 1, &svbuf.buffer, &offset);
                (self.fns.cmd_draw)(cmd, self.static_vert_count, 1, 0, 0);
            }

            // Draw dynamic vertices
            if dyn_vert_count > 0 {
                let dvbuf = self.dynamic_vbuf.as_ref().unwrap();
                let offset: VkDeviceSize = 0;
                (self.fns.cmd_bind_vertex_buffers)(cmd, 0, 1, &dvbuf.buffer, &offset);
                (self.fns.cmd_draw)(cmd, dyn_vert_count, 1, 0, 0);
            }

            (self.fns.cmd_end_render_pass)(cmd);

            // Transition color image for copy
            let barrier = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
                old_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: rt.color_image,
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            (self.fns.cmd_pipeline_barrier)(
                cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, ptr::null(), 0, ptr::null(),
                1, &barrier as *const VkImageMemoryBarrier as *const c_void,
            );

            // Copy to current frame's readback buffer
            let region = VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource_layers: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset_x: 0,
                image_offset_y: 0,
                image_offset_z: 0,
                image_extent: VkExtent3D { width, height, depth: 1 },
            };
            (self.fns.cmd_copy_image_to_buffer)(
                cmd, rt.color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                rt.readback_bufs[curr].buffer, 1, &region,
            );

            // Barrier: transfer → host read
            let buf_barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
                dst_access_mask: VK_ACCESS_HOST_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: rt.readback_bufs[curr].buffer,
                offset: 0,
                size: VK_WHOLE_SIZE,
            };
            (self.fns.cmd_pipeline_barrier)(
                cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, ptr::null(),
                1, &buf_barrier,
                0, ptr::null(),
            );

            (self.fns.end_cmd_buf)(cmd);

            // Submit with current fence (non-blocking)
            (self.fns.reset_fences)(self.device, 1, &self.gfx_fences[curr]);
            let submit = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &self.gfx_cmd_bufs[curr],
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            (self.fns.queue_submit)(self.queue, 1, &submit, self.gfx_fences[curr]);

            // Advance frame index
            self.gfx_frame_idx = prev;
            self.gfx_has_prev_frame = true;
        }
    }

    pub fn has_graphics(&self) -> bool {
        self.gfx_pipeline.is_some() && self.render_target.is_some()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            (self.fns.device_wait_idle)(self.device);
            // Wait for any in-flight graphics work
            (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[0], 1, u64::MAX);
            (self.fns.wait_for_fences)(self.device, 1, &self.gfx_fences[1], 1, u64::MAX);
            (self.fns.destroy_fence)(self.device, self.gfx_fences[0], ptr::null());
            (self.fns.destroy_fence)(self.device, self.gfx_fences[1], ptr::null());
            // Graphics resources
            if let Some(ref vbuf) = self.static_vbuf {
                (self.fns.unmap_mem)(self.device, vbuf.memory);
                (self.fns.destroy_buffer)(self.device, vbuf.buffer, ptr::null());
                (self.fns.free_mem)(self.device, vbuf.memory, ptr::null());
            }
            if let Some(ref vbuf) = self.dynamic_vbuf {
                (self.fns.unmap_mem)(self.device, vbuf.memory);
                (self.fns.destroy_buffer)(self.device, vbuf.buffer, ptr::null());
                (self.fns.free_mem)(self.device, vbuf.memory, ptr::null());
            }
            if let Some(ref gfx) = self.gfx_pipeline {
                (self.fns.destroy_pipeline)(self.device, gfx.pipeline, ptr::null());
                (self.fns.destroy_pipeline_layout)(self.device, gfx.layout, ptr::null());
            }
            if let Some(ref rt) = self.render_target {
                let rp = rt.render_pass;
                (self.fns.destroy_framebuffer)(self.device, rt.framebuffer, ptr::null());
                (self.fns.destroy_image_view)(self.device, rt.color_view, ptr::null());
                (self.fns.destroy_image_view)(self.device, rt.depth_view, ptr::null());
                (self.fns.destroy_image)(self.device, rt.color_image, ptr::null());
                (self.fns.free_mem)(self.device, rt.color_memory, ptr::null());
                (self.fns.destroy_image)(self.device, rt.depth_image, ptr::null());
                (self.fns.free_mem)(self.device, rt.depth_memory, ptr::null());
                for rb in &rt.readback_bufs {
                    (self.fns.unmap_mem)(self.device, rb.memory);
                    (self.fns.destroy_buffer)(self.device, rb.buffer, ptr::null());
                    (self.fns.free_mem)(self.device, rb.memory, ptr::null());
                }
                (self.fns.destroy_render_pass)(self.device, rp, ptr::null());
            }
            // Compute resources
            for (_, pipe) in &self.pipelines {
                (self.fns.destroy_pipeline)(self.device, pipe.pipeline, ptr::null());
                (self.fns.destroy_pipeline_layout)(self.device, pipe.layout, ptr::null());
                (self.fns.destroy_desc_set_layout)(self.device, pipe.desc_set_layout, ptr::null());
            }
            (self.fns.destroy_desc_pool)(self.device, self.desc_pool, ptr::null());
            (self.fns.destroy_fence)(self.device, self.fence, ptr::null());
            (self.fns.destroy_cmd_pool)(self.device, self.cmd_pool, ptr::null());
            (self.fns.destroy_device)(self.device, ptr::null());
            (self.fns.destroy_instance)(self.instance, ptr::null());
        }
    }
}
