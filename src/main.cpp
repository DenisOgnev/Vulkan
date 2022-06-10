#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>
#include <array>
#include <chrono>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::string NAME = "Vulkan";


const int MAX_FRAMES_IN_FLIGHT = 2;
uint32_t current_frame = 0;


const std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };


const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };


#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif


VkResult create_debug_utils_messenger_EXT(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
	const VkAllocationCallbacks* p_allocator, VkDebugUtilsMessengerEXT* p_debug_messenger) 
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
		return func(instance, p_create_info, p_allocator, p_debug_messenger);
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroy_debug_utils_manager_EXT(VkInstance instance, VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks* p_allocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
		func(instance, debug_messenger, p_allocator);
}


struct QueueFamiliesIndices
{
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool is_complete()
	{
		return graphics_family.has_value() && present_family.has_value();
	}
};


struct SwapchainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};


struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription get_binding_description()
	{
		VkVertexInputBindingDescription binding_descripstion{};
		binding_descripstion.binding = 0;
		binding_descripstion.stride = sizeof(Vertex);
		binding_descripstion.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return binding_descripstion;
	}

	static std::array<VkVertexInputAttributeDescription, 2> get_attribute_descriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{};

		attribute_descriptions[0].binding = 0;
		attribute_descriptions[0].location = 0;
		attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[0].offset = offsetof(Vertex, pos);

		attribute_descriptions[1].binding = 0;
		attribute_descriptions[1].location = 1;
		attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[1].offset = offsetof(Vertex, color);

		return attribute_descriptions;
	}
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
};

const std::vector<Vertex> vertices =
{
	{{-0.5f, -0.5f,  0.0f}, {1.0f, 0.0f, 0.0f}},
	{{ 0.5f, -0.5f,  0.0f}, {0.0f, 1.0f, 0.0f}},
	{{ 0.5f,  0.5f,  0.0f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f,  0.5f,  0.0f}, {1.0f, 0.0f, 1.0f}}
};

const std::vector<uint16_t> indices =
{
	0, 1, 2, 2, 3, 0
};

class VkApp
{
public:
	void run()
	{
		init_window();
		init_vulkan();
		main_loop();
		cleanup();
	}

private:
	GLFWwindow* window;

	// Vulkan
	VkInstance instance;
	VkDebugUtilsMessengerEXT debug_messenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphics_queue;
	VkQueue present_queue;

	VkSwapchainKHR swapchain;
	VkFormat swapchain_image_format;
	VkExtent2D swapchain_extent;
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	std::vector<VkFramebuffer> swapchain_framebuffers;

	VkRenderPass render_pass;
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphics_pipeline;

	VkCommandPool command_pool;
	std::vector<VkCommandBuffer> command_buffers;

	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;

	bool framebuffer_resized = false;

	VkBuffer vertex_buffer;
	VkDeviceMemory vertex_buffer_memory;

	VkBuffer index_buffer;
	VkDeviceMemory index_buffer_memory;

	std::vector<VkBuffer> uniform_buffers;
	std::vector<VkDeviceMemory> uniform_buffers_memory;

	void init_window()
	{
		if (!glfwInit())
		{
			glfwTerminate();
			throw std::runtime_error("Failed to initialize GLFW");
		}

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, NAME.c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
	}

	void init_vulkan()
	{
		create_instance();
		setup_debug_messenger();
		create_surface();
		pick_physical_device();
		create_logical_device();
		create_swapchain();
		create_image_views();
		create_render_pass();
		create_descriptor_set_layout();
		create_graphics_pipeline();
		create_framebuffers();
		create_command_pool();
		create_vertex_buffer();
		create_index_buffer();
		create_uniform_buffers();
		create_descriptor_pool();
		create_command_buffers();
		create_sync_objects();
	}

	void main_loop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			draw_frame();
		}

		vkDeviceWaitIdle(device);
	}

	void cleanup()
	{
		cleanup_swapchain();
		
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(device, uniform_buffers[i], nullptr);
			vkFreeMemory(device, uniform_buffers_memory[i], nullptr);
		}

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);

		vkDestroyBuffer(device, vertex_buffer, nullptr);
		vkFreeMemory(device, vertex_buffer_memory, nullptr);
		
		vkDestroyBuffer(device, index_buffer, nullptr);
		vkFreeMemory(device, index_buffer_memory, nullptr);		

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
			vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
			vkDestroyFence(device, in_flight_fences[i], nullptr);
		}

		vkDestroyCommandPool(device, command_pool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enable_validation_layers)
		{
			destroy_debug_utils_manager_EXT(instance, debug_messenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	// Vulkan objects creating
	void create_instance()
	{
		if (enable_validation_layers && !check_validation_layers_support())
			throw std::runtime_error("Validation layers are not available");

		VkApplicationInfo app_info{};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "Vulkan";
		app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		app_info.pEngineName = nullptr;
		app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;

		std::vector<const char*> glfw_extensions = get_required_extensions();

		if (!check_instance_extensions_support(glfw_extensions))
			throw std::runtime_error("Extensions are not available");

		create_info.enabledExtensionCount = static_cast<uint32_t>(glfw_extensions.size());
		create_info.ppEnabledExtensionNames = glfw_extensions.data();

		if (enable_validation_layers)
		{
			VkDebugUtilsMessengerCreateInfoEXT debug_create_info;

			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();

			populate_debug_messenger_create_info(debug_create_info);
			create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info; // useless cast?
		}
		else
		{
			create_info.enabledLayerCount = 0;
			create_info.pNext = nullptr;
		}

		if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("Failed to create instance");
	}

	void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info)
	{
		create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		create_info.messageSeverity =
			//VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
			//VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
		create_info.pfnUserCallback = debug_callback;
		create_info.pUserData = nullptr;
	}

	void setup_debug_messenger()
	{
		if (!enable_validation_layers)
			return;

		VkDebugUtilsMessengerCreateInfoEXT create_info;
		populate_debug_messenger_create_info(create_info);

		if (create_debug_utils_messenger_EXT(instance, &create_info, nullptr, &debug_messenger) != VK_SUCCESS)
			throw std::runtime_error("Failed to set up debug manager");
	}

	void create_surface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("Failed to create window surface");
	}

	void pick_physical_device()
	{
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
		if (device_count == 0)
			throw std::runtime_error("Failed to find GPU with Vulkan support");
		std::vector<VkPhysicalDevice> devices(device_count);
		vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

		for (const auto& device : devices)
		{
			if (is_device_suitable(device))
			{
				physical_device = device;
				break;
			}
		}

		if (physical_device == VK_NULL_HANDLE)
			throw std::runtime_error("Failed to find suitable GPU");
	}

	void create_logical_device()
	{
		QueueFamiliesIndices indices = find_queue_families(physical_device);

		std::set<uint32_t> unique_queue_families = { indices.graphics_family.value(), indices.present_family.value() };

		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		float queue_priority = 1.0f;
		for (uint32_t queue_familiy : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queue_create_info{};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_familiy;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;

			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features{};

		VkDeviceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		create_info.pQueueCreateInfos = queue_create_infos.data();
		create_info.pEnabledFeatures = &device_features;
		create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		create_info.ppEnabledExtensionNames = device_extensions.data();

		if (enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			create_info.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create logical device");
		}
		vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
		vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
	}

	void create_swapchain()
	{
		SwapchainSupportDetails swapchain_support = query_swapchain_support(physical_device);

		VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swapchain_support.formats);
		VkPresentModeKHR present_mode = choose_swap_present_mode(swapchain_support.present_modes);
		VkExtent2D extent = choose_swap_extent(swapchain_support.capabilities);

		swapchain_image_format = surface_format.format;
		swapchain_extent = extent;

		uint32_t image_count = swapchain_support.capabilities.minImageCount + 1;
		if (swapchain_support.capabilities.maxImageCount > 0 && image_count > swapchain_support.capabilities.maxImageCount)
			image_count = swapchain_support.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = surface;
		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamiliesIndices indices = find_queue_families(physical_device);
		std::vector<uint32_t> queue_family_indices = { indices.graphics_family.value(), indices.present_family.value() };

		if (indices.graphics_family != indices.present_family)
		{
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = (uint32_t)queue_family_indices.size();
			create_info.pQueueFamilyIndices = queue_family_indices.data();
		}
		else
		{
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			create_info.queueFamilyIndexCount = 0;
			create_info.pQueueFamilyIndices = nullptr;
		}

		create_info.preTransform = swapchain_support.capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;
		create_info.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain) != VK_SUCCESS)
			throw std::runtime_error("Failed to create swap chain");

		vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
		swapchain_images.resize(image_count);
		vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());
	}

	void create_image_views()
	{
		swapchain_image_views.resize(swapchain_images.size());

		for (size_t i = 0; i < swapchain_image_views.size(); i++)
		{
			VkImageViewCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			create_info.image = swapchain_images[i];
			create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			create_info.format = swapchain_image_format;
			create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			create_info.subresourceRange.baseMipLevel = 0;
			create_info.subresourceRange.levelCount = 1;
			create_info.subresourceRange.baseArrayLayer = 0;
			create_info.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &create_info, nullptr, &swapchain_image_views[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create image views");
		}

	}

	void create_render_pass()
	{
		VkAttachmentDescription color_attachment{};
		color_attachment.format = swapchain_image_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_reference{};
		color_attachment_reference.attachment = 0;
		color_attachment_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass_description{};
		subpass_description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_description.colorAttachmentCount = 1;
		subpass_description.pColorAttachments = &color_attachment_reference;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo render_pass_create_info{};
		render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount = 1;
		render_pass_create_info.pAttachments = &color_attachment;
		render_pass_create_info.subpassCount = 1;
		render_pass_create_info.pSubpasses = &subpass_description;
		render_pass_create_info.dependencyCount = 1;
		render_pass_create_info.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &render_pass_create_info, nullptr, &render_pass) != VK_SUCCESS)
			throw std::runtime_error("Failed to create render pass");
	}

	void create_descriptor_set_layout()
	{
		VkDescriptorSetLayoutBinding ubo_layout_binding{};
		ubo_layout_binding.binding = 0;
		ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		ubo_layout_binding.descriptorCount = 1;
		ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		ubo_layout_binding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{};
		descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptor_set_layout_create_info.bindingCount = 1;
		descriptor_set_layout_create_info.pBindings = &ubo_layout_binding;

		if (vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
			throw std::runtime_error("Failed to create descritor set layout");
	}

	void create_graphics_pipeline()
	{
		auto vert_shader_code = read_file("../../src/shaders/vert.spv");
		auto frag_shader_code = read_file("../../src/shaders/frag.spv");

		VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
		VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

		VkPipelineShaderStageCreateInfo vert_shader_create_info{};
		vert_shader_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_create_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_create_info.module = vert_shader_module;
		vert_shader_create_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_shader_create_info{};
		frag_shader_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_create_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_create_info.module = frag_shader_module;
		frag_shader_create_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[] = { vert_shader_create_info, frag_shader_create_info };

		auto binding_descriptions = Vertex::get_binding_description();
		auto attribute_descriptions = Vertex::get_attribute_descriptions();

		VkPipelineVertexInputStateCreateInfo vertex_input_create_info{};
		vertex_input_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_create_info.vertexBindingDescriptionCount = 1;
		vertex_input_create_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
		vertex_input_create_info.pVertexBindingDescriptions = &binding_descriptions;
		vertex_input_create_info.pVertexAttributeDescriptions = attribute_descriptions.data();

		VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info{};
		input_assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly_create_info.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapchain_extent.width;
		viewport.height = (float)swapchain_extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapchain_extent;

		VkPipelineViewportStateCreateInfo viewport_state_create_info{};
		viewport_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state_create_info.viewportCount = 1;
		viewport_state_create_info.pViewports = &viewport;
		viewport_state_create_info.scissorCount = 1;
		viewport_state_create_info.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterization_state_create_info{};
		rasterization_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_state_create_info.depthClampEnable = VK_FALSE;
		rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_state_create_info.lineWidth = 1.0f;
		rasterization_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterization_state_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_state_create_info.depthBiasEnable = VK_FALSE;
		rasterization_state_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_state_create_info.depthBiasClamp = 0.0f;
		rasterization_state_create_info.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisample_state_create_info{};
		multisample_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_state_create_info.sampleShadingEnable = VK_FALSE;
		multisample_state_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_state_create_info.minSampleShading = 1.0f;
		multisample_state_create_info.pSampleMask = nullptr;
		multisample_state_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_state_create_info.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState color_blend_attachment_state{};
		color_blend_attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo color_blend_state_create_info{};
		color_blend_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_state_create_info.logicOpEnable = VK_FALSE;
		color_blend_state_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_state_create_info.attachmentCount = 1;
		color_blend_state_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_state_create_info.blendConstants[0] = 0.0f;
		color_blend_state_create_info.blendConstants[1] = 0.0f;
		color_blend_state_create_info.blendConstants[2] = 0.0f;
		color_blend_state_create_info.blendConstants[3] = 0.0f;

		std::vector<VkDynamicState> dynamic_states =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_LINE_WIDTH
		};

		VkPipelineDynamicStateCreateInfo dynamic_state_create_info{};
		dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_state_create_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic_state_create_info.pDynamicStates = dynamic_states.data();

		VkPipelineLayoutCreateInfo pipeline_layout_create_info{};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &descriptor_set_layout;
		pipeline_layout_create_info.pushConstantRangeCount = 0;
		pipeline_layout_create_info.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS)
			throw std::runtime_error("Failed to create pipeline layout");

		VkGraphicsPipelineCreateInfo pipeline_create_info{};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = shader_stages;
		pipeline_create_info.pVertexInputState = &vertex_input_create_info;
		pipeline_create_info.pInputAssemblyState = &input_assembly_create_info;
		pipeline_create_info.pViewportState = &viewport_state_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
		pipeline_create_info.pMultisampleState = &multisample_state_create_info;
		pipeline_create_info.pDepthStencilState = nullptr;
		pipeline_create_info.pColorBlendState = &color_blend_state_create_info;
		pipeline_create_info.pDynamicState = nullptr;
		pipeline_create_info.layout = pipeline_layout;
		pipeline_create_info.renderPass = render_pass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		pipeline_create_info.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline) != VK_SUCCESS)
			throw std::runtime_error("Failed to create graphics pipeline");

		vkDestroyShaderModule(device, frag_shader_module, nullptr);
		vkDestroyShaderModule(device, vert_shader_module, nullptr);
	}

	void create_framebuffers()
	{
		swapchain_framebuffers.resize(swapchain_image_views.size());

		for (size_t i = 0; i < swapchain_image_views.size(); i++)
		{
			VkImageView attachments[] =
			{
				swapchain_image_views[i]
			};

			VkFramebufferCreateInfo framebuffer_create_info{};
			framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer_create_info.renderPass = render_pass;
			framebuffer_create_info.attachmentCount = 1;
			framebuffer_create_info.pAttachments = attachments;
			framebuffer_create_info.width = swapchain_extent.width;
			framebuffer_create_info.height = swapchain_extent.height;
			framebuffer_create_info.layers = 1;

			if (vkCreateFramebuffer(device, &framebuffer_create_info, nullptr, &swapchain_framebuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create framebuffer");
		}
	}

	void create_command_pool()
	{
		QueueFamiliesIndices queue_family_indices = find_queue_families(physical_device);

		VkCommandPoolCreateInfo command_pool_create_info{};
		command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		command_pool_create_info.queueFamilyIndex = queue_family_indices.graphics_family.value();

		if (vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool) != VK_SUCCESS)
			throw std::runtime_error("Failed to create command pool");
	}

	void create_command_buffers()
	{
		command_buffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo command_buffer_allocate_info{};
		command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_allocate_info.commandPool = command_pool;
		command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_allocate_info.commandBufferCount = (uint32_t)command_buffers.size();

		if (vkAllocateCommandBuffers(device, &command_buffer_allocate_info, command_buffers.data()) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate command buffers");
	}

	void create_sync_objects()
	{
		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphore_create_info{};
		semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(device, &semaphore_create_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphore_create_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fence_create_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create sync objects");
		}
	}

	void create_vertex_buffer()
	{
		VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void* data;
		vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, vertices.data(), (size_t)buffer_size);
		vkUnmapMemory(device, staging_buffer_memory);

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_memory);

		copy_buffer(staging_buffer, vertex_buffer, buffer_size);

		vkDestroyBuffer(device, staging_buffer, nullptr);
		vkFreeMemory(device, staging_buffer_memory, nullptr);
	}

	void create_index_buffer()
	{
		VkDeviceSize buffer_size = sizeof(indices[0]) * indices.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void* data;
		vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, indices.data(), (size_t)buffer_size);
		vkUnmapMemory(device, staging_buffer_memory);

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer, index_buffer_memory);

		copy_buffer(staging_buffer, index_buffer, buffer_size);

		vkDestroyBuffer(device, staging_buffer, nullptr);
		vkFreeMemory(device, staging_buffer_memory, nullptr);
	}

	void create_uniform_buffers()
	{
		VkDeviceSize buffer_size = sizeof(UniformBufferObject);

		uniform_buffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniform_buffers_memory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			create_buffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniform_buffers[i], uniform_buffers_memory[i]);
		}
	}

	void create_descriptor_pool()
	{
	}

	// Utils
	void draw_frame()
	{
		vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		{
			recreate_swapchain();
			return;
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to acquire swapchain image");
		}

		vkResetFences(device, 1, &in_flight_fences[current_frame]);

		vkResetCommandBuffer(command_buffers[current_frame], 0);
		record_command_buffer(command_buffers[current_frame], image_index);

		update_uniform_buffer(current_frame);

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore wait_semaphores[] = { image_available_semaphores[current_frame] };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[current_frame];

		VkSemaphore signal_semaphores[] = { render_finished_semaphores[current_frame] };
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		if (vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]) != VK_SUCCESS)
			throw std::runtime_error("Failed to submit draw command buffer");

		VkPresentInfoKHR present_info{};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		VkSwapchainKHR swapchains[] = { swapchain };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapchains;
		present_info.pImageIndices = &image_index;
		present_info.pResults = nullptr;

		result = vkQueuePresentKHR(present_queue, &present_info);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
		{
			framebuffer_resized = false;
			recreate_swapchain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present swapchain image");
		}

		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void update_uniform_buffer(uint32_t current_image)
	{
		static auto start_time = std::chrono::high_resolution_clock::now();

		auto current_time = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view =  glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.projection = glm::perspective(glm::radians(45.0f), swapchain_extent.width / (float)swapchain_extent.height, 0.1f, 10.0f);
		ubo.projection[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniform_buffers_memory[current_image]);
	}

	void record_command_buffer(VkCommandBuffer command_buffer, uint32_t image_index)
	{
		VkCommandBufferBeginInfo command_buffer_begin_info{};
		command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		command_buffer_begin_info.flags = 0;
		command_buffer_begin_info.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) != VK_SUCCESS)
			throw std::runtime_error("Failed to begin recording command buffer");

		VkRenderPassBeginInfo render_pass_begin_info{};
		render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_begin_info.renderPass = render_pass;
		render_pass_begin_info.framebuffer = swapchain_framebuffers[image_index];
		render_pass_begin_info.renderArea.offset = { 0, 0 };
		render_pass_begin_info.renderArea.extent = swapchain_extent;

		VkClearValue clear_color = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
		render_pass_begin_info.clearValueCount = 1;
		render_pass_begin_info.pClearValues = &clear_color;

		vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkBuffer vertex_buffers[] = { vertex_buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

		vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

		//vkCmdDraw(command_buffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
		vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(command_buffer);

		if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to record command buffer");
	}

	void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &buffer_memory)
	{
		VkBufferCreateInfo buffer_create_info{};
		buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_create_info.size = size;
		buffer_create_info.usage = usage;
		buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to create vertex buffer");

		VkMemoryRequirements mem_requirements;
		vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

		VkMemoryAllocateInfo allocate_info{};
		allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate_info.allocationSize = mem_requirements.size;
		allocate_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocate_info, nullptr, &buffer_memory) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate vertex buffer memory");

		vkBindBufferMemory(device, buffer, buffer_memory, 0);
	}

	void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
	{
		VkCommandBufferAllocateInfo command_buffer_allocate_info{};
		command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_allocate_info.commandPool = command_pool;
		command_buffer_allocate_info.commandBufferCount = 1;

		VkCommandBuffer command_buffer;
		if (vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate command buffer");

		VkCommandBufferBeginInfo command_buffer_begin_info{};
		command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);

		VkBufferCopy copy_region{};
		copy_region.srcOffset = 0;
		copy_region.dstOffset = 0;
		copy_region.size = size;
		vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);

		vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
	}

	void cleanup_swapchain()
	{
		for (const auto& swapchain_framebuffer : swapchain_framebuffers)
		{
			vkDestroyFramebuffer(device, swapchain_framebuffer, nullptr);
		}

		vkDestroyPipeline(device, graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyRenderPass(device, render_pass, nullptr);

		for (const auto& swapchain_image_view : swapchain_image_views)
		{
			vkDestroyImageView(device, swapchain_image_view, nullptr);
		}

		vkDestroySwapchainKHR(device, swapchain, nullptr);
	}

	void recreate_swapchain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanup_swapchain();

		create_swapchain();
		create_image_views();
		create_render_pass();
		create_graphics_pipeline();
		create_framebuffers();
	}

	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
		{
			if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}

		throw std::runtime_error("Failed to find suitable memory type");
	}

	bool is_device_suitable(VkPhysicalDevice device)
	{
		QueueFamiliesIndices indices = find_queue_families(device);

		bool extensions_supported = check_device_extensions_support(device);

		bool swapchain_adequate = false;
		if (extensions_supported)
		{
			SwapchainSupportDetails swapchain_support = query_swapchain_support(device);
			swapchain_adequate = !swapchain_support.formats.empty() && !swapchain_support.present_modes.empty();
		}

		return indices.is_complete() && extensions_supported && swapchain_adequate;
	}

	QueueFamiliesIndices find_queue_families(VkPhysicalDevice device)
	{
		QueueFamiliesIndices indices;

		uint32_t queue_families_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_families_count, nullptr);
		std::vector<VkQueueFamilyProperties> queue_families(queue_families_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_families_count, queue_families.data());

		for (int i = 0; i < queue_families.size(); i++)
		{
			if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphics_family = i;

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
			if (present_support)
				indices.present_family = i;

			if (indices.is_complete())
				break;
		}

		return indices;
	}

	bool check_device_extensions_support(VkPhysicalDevice device)
	{
		uint32_t extensions_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensions_count, nullptr);
		std::vector<VkExtensionProperties> available_extensions(extensions_count);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensions_count, available_extensions.data());

		std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

		for (const auto& extension : available_extensions)
		{
			required_extensions.erase(extension.extensionName);
		}

		return required_extensions.empty();
	}

	bool check_instance_extensions_support(const std::vector<const char*>& glfw_extensions)
	{
		uint32_t extensions_count = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
		std::vector<VkExtensionProperties> extensions(extensions_count);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, extensions.data());

		for (const auto& glfw_extension_name : glfw_extensions)
		{
			bool extension_found = false;
			for (const auto& extension_properties : extensions)
			{
				if (strcmp(glfw_extension_name, extension_properties.extensionName) == 0)
				{
					extension_found = true;
					break;
				}
			}
			if (!extension_found)
				return false;
		}

		return true;
	}

	bool check_validation_layers_support()
	{
		uint32_t layers_count = 0;
		vkEnumerateInstanceLayerProperties(&layers_count, nullptr);
		std::vector<VkLayerProperties> available_layers(layers_count);
		vkEnumerateInstanceLayerProperties(&layers_count, available_layers.data());

		for (const char* layer_name : validation_layers)
		{
			bool layer_found = false;
			for (const auto& layer_properties : available_layers)
			{
				if (strcmp(layer_name, layer_properties.layerName) == 0)
				{
					layer_found = true;
					break;
				}
			}
			if (!layer_found)
				return false;
		}

		return true;
	}

	std::vector<const char*> get_required_extensions()
	{
		uint32_t glfw_extensions_count = 0;
		const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensions_count);

		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extensions_count);

		if (enable_validation_layers)
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		return extensions;
	}

	SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device)
	{
		SwapchainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
		if (format_count != 0)
		{
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
		}

		uint32_t present_modes_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_modes_count, nullptr);
		if (present_modes_count != 0)
		{
			details.present_modes.resize(present_modes_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_modes_count, details.present_modes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
	{
		for (const auto& available_format : available_formats)
		{
			if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return available_format;
		}

		return available_formats[0];
	}

	VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes)
	{
		for (const auto& available_present_mode : available_present_modes)
		{
			if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
				return available_present_mode;
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return capabilities.currentExtent;

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actual_extend =
		{
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actual_extend.width = std::clamp(actual_extend.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actual_extend.height = std::clamp(actual_extend.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actual_extend;
	}

	static std::vector<char> read_file(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("Failed to open file");

		size_t file_size = (size_t)file.tellg();
		std::vector<char> buffer(file_size);

		file.seekg(0);
		file.read(buffer.data(), file_size);

		file.close();

		return buffer;
	}

	VkShaderModule create_shader_module(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shader_module;
		if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
			throw std::runtime_error("Failed to create shader module");

		return shader_module;
	}

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<VkApp*>(glfwGetWindowUserPointer(window));
		app->framebuffer_resized = true;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
		void* p_user_data)
	{
		std::cerr << "Validation layer: " << p_callback_data->pMessage << "\n";

		return VK_FALSE;
	}
};

int main()
{
	VkApp app;
	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}

	return 0;
}