// Vulkan compute: GPU-accelerated game simulation
// dlopen libvulkan.so.1, create compute pipeline, dispatch SPIR-V shaders

#![allow(unsafe_op_in_unsafe_fn, non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};
use std::ptr;

use crate::gpu_kernels;

// --- libc ---
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_LAZY: c_int = 1;

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

// --- Vulkan constants ---
const VK_SUCCESS: i32 = 0;
const VK_QUEUE_COMPUTE_BIT: u32 = 2;
const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
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
}

// --- GPU buffer ---

pub struct GpuBuf {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    mapped: *mut c_void,
    pub size: usize,
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
                if qf.queue_flags & VK_QUEUE_COMPUTE_BIT != 0 {
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

        // Create fence
        let fence_info = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
        };
        let mut fence: VkFence = 0;
        (fns.create_fence)(device, &fence_info, ptr::null(), &mut fence);

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
        };

        // Build pipelines
        ctx.add_pipeline("test_multiply", &gpu_kernels::build_test_multiply(), 1, 4)?;
        ctx.add_pipeline("particle_update", &gpu_kernels::build_particle_update(), 7, 12)?;

        Ok(ctx)
    }

    unsafe fn add_pipeline(&mut self, name: &'static str, spirv: &[u32], binding_count: u32, push_size: u32) -> Result<(), &'static str> {
        // Create shader module
        let module_info = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
        };
        let mut module: VkShaderModule = 0;
        let res = (self.fns.create_shader_module)(self.device, &module_info, ptr::null(), &mut module);
        if res != VK_SUCCESS { return Err("vkCreateShaderModule failed"); }

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
        unsafe {
            let buf_info = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                size: size_bytes as u64,
                usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
            };
            let mut buffer: VkBuffer = 0;
            (self.fns.create_buffer)(self.device, &buf_info, ptr::null(), &mut buffer);

            let mut reqs = std::mem::zeroed::<VkMemoryRequirements>();
            (self.fns.get_buf_mem_reqs)(self.device, buffer, &mut reqs);

            let mem_type = find_memory_type(
                &self.mem_props, reqs.memory_type_bits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            ).expect("No suitable memory type");

            let alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: reqs.size,
                memory_type_index: mem_type,
            };
            let mut memory: VkDeviceMemory = 0;
            (self.fns.alloc_mem)(self.device, &alloc_info, ptr::null(), &mut memory);
            (self.fns.bind_buf_mem)(self.device, buffer, memory, 0);

            let mut mapped: *mut c_void = ptr::null_mut();
            (self.fns.map_mem)(self.device, memory, 0, VK_WHOLE_SIZE, 0, &mut mapped);

            GpuBuf { buffer, memory, mapped, size: size_bytes }
        }
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
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
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
