import ctypes
import os.path

import numpy as np
import vulkan as vk


def find_memory_type_index(physical_device, memory_property_flags):
    memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device, None)

    for i in range(memory_properties.memoryTypeCount):
        if (memory_properties.memoryTypes[i].propertyFlags & memory_property_flags) == memory_property_flags:
            return i

    raise Exception("Failed to find suitable memory type index")


def create_shader_module(device, filepath):
    shader_code = open(filepath, "rb").read()
    create_info = vk.VkShaderModuleCreateInfo(
        codeSize=len(shader_code),
        pCode=shader_code
    )
    return vk.vkCreateShaderModule(device, create_info, None)


class Image:
    def __init__(self):
        self.device = None
        self.image = None
        self.memory = None
        self.image_view = None
        self.format = None

    def create(self, device, physical_device, image_create_info, image_view_create_info):
        self.destroy()
        self.device = device
        self.image = vk.vkCreateImage(self.device, image_create_info, None)

        memory_requirements = vk.vkGetImageMemoryRequirements(self.device, self.image, None)
        memory_property_flags = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        memory_type_index = find_memory_type_index(physical_device, memory_property_flags)

        allocate_info = vk.VkMemoryAllocateInfo(
            allocationSize=memory_requirements.size,
            memoryTypeIndex=memory_type_index
        )
        self.memory = vk.vkAllocateMemory(device, allocate_info, None)
        vk.vkBindImageMemory(self.device, self.image, self.memory, 0)

        image_view_create_info.image = self.image
        self.image_view = vk.vkCreateImageView(device, image_view_create_info, None)

        self.format = image_view_create_info.format

    def create_color_image(self, device, physical_device, extent, format, usage):
        image_type = vk.VK_IMAGE_TYPE_3D if extent.depth > 1 \
            else (vk.VK_IMAGE_TYPE_2D if extent.height > 1 else vk.VK_IMAGE_TYPE_1D)

        image_create_info = vk.VkImageCreateInfo(
            imageType=image_type,
            format=format,
            extent=extent,
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )

        view_type = vk.VK_IMAGE_VIEW_TYPE_3D if image_type == vk.VK_IMAGE_TYPE_3D \
            else (vk.VK_IMAGE_VIEW_TYPE_2D if image_type == vk.VK_IMAGE_TYPE_2D else vk.VK_IMAGE_VIEW_TYPE_1D)

        components = vk.VkComponentMapping(
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY
        )
        subresource_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        image_view_create_info = vk.VkImageViewCreateInfo(
            viewType=view_type,
            format=format,
            components=components,
            subresourceRange=subresource_range
        )

        self.create(device, physical_device, image_create_info, image_view_create_info)

    def destroy(self):
        if self.device is not None:
            vk.vkDestroyImage(self.device, self.image, None)
            vk.vkFreeMemory(self.device, self.memory, None)
            vk.vkDestroyImageView(self.device, self.image_view, None)

            self.device = None
            self.image = None
            self.memory = None
            self.image_view = None


class Buffer:
    def __init__(self):
        self.device = None
        self.buffer = None
        self.memory = None
        self.memory_size = None

    def _create(self, device, physical_device, buffer_create_info, memory_property_flags):
        self.destroy()

        self.device = device
        self.buffer = vk.vkCreateBuffer(device, buffer_create_info, None)

        memory_requirements = vk.vkGetBufferMemoryRequirements(self.device, self.buffer, None)
        memory_type_index = find_memory_type_index(physical_device, memory_property_flags)

        allocate_info = vk.VkMemoryAllocateInfo(
            allocationSize=memory_requirements.size,
            memoryTypeIndex=memory_type_index
        )
        self.memory = vk.vkAllocateMemory(device, allocate_info, None)
        self.memory_size = allocate_info.allocationSize
        vk.vkBindBufferMemory(self.device, self.buffer, self.memory, 0)

    def destroy(self):
        if self.device is not None:
            vk.vkDestroyBuffer(self.device, self.buffer, None)
            vk.vkFreeMemory(self.device, self.memory, None)

            self.device = None
            self.buffer = None
            self.memory = None
            self.memory_size = None

    def map_memory(self):
        return vk.vkMapMemory(self.device, self.memory, 0, self.memory_size, 0)

    def unmap_memory(self):
        vk.vkUnmapMemory(self.device, self.memory)

    @staticmethod
    def create(device, physical_device, size, usage, memory_property_flags):
        create_info = vk.VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = Buffer()
        buffer._create(device, physical_device, create_info, memory_property_flags)
        return buffer


class PushConstants(ctypes.Structure):
    _fields_ = [
        ("pointCount", ctypes.c_uint32),
        ("kernelRadius", ctypes.c_uint32),
    ]


class Regularizer:
    def __init__(self):
        self.shader_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
        self.texture_size = 1024

        # Create instance
        application_info = vk.VkApplicationInfo(
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0)
        )
        enabled_layers = ['VK_LAYER_KHRONOS_validation']
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=application_info,
            enabledLayerCount=1,
            ppEnabledLayerNames=enabled_layers
        )
        self.instance = vk.vkCreateInstance(create_info, None)

        # Pick physical device
        queue_flags = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT
        self.physical_device = None

        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        for physical_device in physical_devices:
            queue_family_properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
            for (queue_family_index, properties) in enumerate(queue_family_properties):
                if (properties.queueFlags & queue_flags) == queue_flags:
                    self.physical_device = physical_device
                    self.queue_family_index = queue_family_index
                    break

            if self.physical_device is not None:
                break

        if self.physical_device is None:
            raise Exception("Failed to find suitable physical device")

        # Create device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        create_info = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info]
        )
        self.device = vk.vkCreateDevice(self.physical_device, create_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family_index, 0)

        # Create command pool
        create_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family_index
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, create_info, None)

        # Allocate command buffer
        allocate_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, allocate_info, None)[0]

        # Create pipline cache
        create_info = vk.VkPipelineCacheCreateInfo()
        self.pipeline_cache = vk.vkCreatePipelineCache(self.device, create_info, None)

        # Create images
        self.images_density = Image()
        self.images_integral_columns = Image()
        self.images_integral_image = Image()
        self.images_upper_left_integral_triangle = Image()
        self.images_upper_right_integral_triangle = Image()
        self.images_deformation = Image()

        image_extent = vk.VkExtent3D(self.texture_size, self.texture_size, 1)
        self.images_density.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        self.images_integral_columns.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32G32_SFLOAT,
            vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        self.images_integral_image.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        self.images_upper_left_integral_triangle.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32G32_SFLOAT,
            vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        self.images_upper_right_integral_triangle.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32G32_SFLOAT,
            vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        self.images_deformation.create_color_image(
            self.device, self.physical_device, image_extent,
            vk.VK_FORMAT_R32G32_SFLOAT,
            vk.VK_IMAGE_USAGE_STORAGE_BIT
        )

        image_subresource_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        image_memory_barriers = [vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_NONE,
            dstAccessMask=vk.VK_ACCESS_NONE,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            subresourceRange=image_subresource_range
        ) for _ in range(6)]
        image_memory_barriers[0].image = self.images_density.image
        image_memory_barriers[1].image = self.images_integral_columns.image
        image_memory_barriers[2].image = self.images_integral_image.image
        image_memory_barriers[3].image = self.images_upper_left_integral_triangle.image
        image_memory_barriers[4].image = self.images_upper_right_integral_triangle.image
        image_memory_barriers[5].image = self.images_deformation.image

        begin_info = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)

        vk.vkResetCommandBuffer(self.command_buffer, 0)
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            srcStageMask=vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
            memoryBarrierCount=0,
            pMemoryBarriers=None,
            bufferMemoryBarrierCount=0,
            pBufferMemoryBarriers=None,
            imageMemoryBarrierCount=len(image_memory_barriers),
            pImageMemoryBarriers=image_memory_barriers
        )
        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer]
        )
        vk.vkQueueSubmit(self.queue, submitCount=1, pSubmits=[submit_info], fence=None)
        vk.vkQueueWaitIdle(self.queue)

        # Create render pass
        attachments = [
            vk.VkAttachmentDescription(
                flags=0,
                format=self.images_density.format,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                finalLayout=vk.VK_IMAGE_LAYOUT_GENERAL
            )
        ]
        color_attachments = [
            vk.VkAttachmentReference(0, vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        ]
        subpasses = [
            vk.VkSubpassDescription(
                flags=0,
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                inputAttachmentCount=0,
                pInputAttachments=None,
                colorAttachmentCount=len(color_attachments),
                pColorAttachments=color_attachments,
                pResolveAttachments=None,
                pDepthStencilAttachment=None,
                preserveAttachmentCount=0,
                pPreserveAttachments=None
            )
        ]
        subpass_dependencies = [
            vk.VkSubpassDependency(
                srcSubpass=0,
                dstSubpass=vk.VK_SUBPASS_EXTERNAL,
                srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT
            )
        ]
        create_info = vk.VkRenderPassCreateInfo(
            attachmentCount=len(attachments),
            pAttachments=attachments,
            subpassCount=len(subpasses),
            pSubpasses=subpasses,
            dependencyCount=len(subpass_dependencies),
            pDependencies=subpass_dependencies
        )
        self.render_pass = vk.vkCreateRenderPass(self.device, create_info, None)

        # Create framebuffer
        framebuffer_attachments = [self.images_density.image_view]
        create_info = vk.VkFramebufferCreateInfo(
            renderPass=self.render_pass,
            attachmentCount=len(framebuffer_attachments),
            pAttachments=framebuffer_attachments,
            width=self.texture_size,
            height=self.texture_size,
            layers=1
        )
        self.framebuffer = vk.vkCreateFramebuffer(self.device, create_info, None)

        # Create sampler
        create_info = vk.VkSamplerCreateInfo(
            magFilter=vk.VK_FILTER_NEAREST,
            minFilter=vk.VK_FILTER_NEAREST,
            mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
            addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            mipLodBias=0.0,
            anisotropyEnable=False,
            maxAnisotropy=0.0,
            compareEnable=False,
            compareOp=vk.VK_COMPARE_OP_NEVER,
            minLod=0.0,
            maxLod=1.0,
            borderColor=vk.VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            unnormalizedCoordinates=False
        )
        self.sampler = vk.vkCreateSampler(self.device, create_info, None)

        # Create descriptor set layout
        bindings = [
            vk.VkDescriptorSetLayoutBinding(0, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                            vk.VK_SHADER_STAGE_VERTEX_BIT | vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(1, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(2, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(3, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(4, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(5, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(6, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT)
        ]
        create_info = vk.VkDescriptorSetLayoutCreateInfo(bindingCount=len(bindings), pBindings=bindings)
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, create_info, None)

        # Create descriptor pool
        descriptor_pool_sizes = [
            vk.VkDescriptorPoolSize(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
            vk.VkDescriptorPoolSize(vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6)
        ]
        create_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=2, poolSizeCount=len(descriptor_pool_sizes), pPoolSizes=descriptor_pool_sizes)
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, create_info, None)

        # Allocate descriptor set
        allocate_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(self.device, allocate_info, None)[0]

        # Create pipeline layout
        push_constant_ranges = [
            vk.VkPushConstantRange(vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, ctypes.sizeof(PushConstants))
        ]
        create_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=len(push_constant_ranges),
            pPushConstantRanges=push_constant_ranges
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, create_info, None)

        # Create pipelines
        shader_modules = [
            create_shader_module(self.device, os.path.join(self.shader_directory, "accumulation.vert.spv")),
            create_shader_module(self.device, os.path.join(self.shader_directory, "accumulation.frag.spv"))
        ]
        shader_stage_create_infos = [
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=shader_modules[0], pName="main", pSpecializationInfo=None
            ),
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=shader_modules[1], pName="main", pSpecializationInfo=None
            )
        ]
        vertex_input_state_create_info = vk.VkPipelineVertexInputStateCreateInfo()
        input_assembly_state_create_info = vk.VkPipelineInputAssemblyStateCreateInfo()
        viewport = vk.VkViewport(0.0, 0.0, self.texture_size, self.texture_size, 0.0, 1.0)
        scissor = vk.VkRect2D(vk.VkOffset2D(0, 0), vk.VkExtent2D(self.texture_size, self.texture_size))
        viewport_state_create_info = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1, pViewports=[viewport], scissorCount=1, pScissors=[scissor]
        )
        rasterization_state_create_info = vk.VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=False,
            depthBiasConstantFactor=0.0,
            depthBiasClamp=0.0,
            depthBiasSlopeFactor=0.0,
            lineWidth=1.0
        )
        multisample_state_create_info = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=False,
            minSampleShading=0.0,
            pSampleMask=None,
            alphaToCoverageEnable=False,
            alphaToOneEnable=False
        )
        color_blend_attachment_states = [
            vk.VkPipelineColorBlendAttachmentState(
                blendEnable=True,
                srcColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                colorBlendOp=vk.VK_BLEND_OP_ADD,
                srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                alphaBlendOp=vk.VK_BLEND_OP_ADD,
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            )
        ]
        color_blend_state_create_info = vk.VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=False,
            logicOp=vk.VK_LOGIC_OP_NO_OP,
            attachmentCount=len(color_blend_attachment_states),
            pAttachments=color_blend_attachment_states,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )
        create_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=len(shader_stage_create_infos),
            pStages=shader_stage_create_infos,
            pVertexInputState=vertex_input_state_create_info,
            pInputAssemblyState=input_assembly_state_create_info,
            pTessellationState=None,
            pViewportState=viewport_state_create_info,
            pRasterizationState=rasterization_state_create_info,
            pMultisampleState=multisample_state_create_info,
            pDepthStencilState=None,
            pColorBlendState=color_blend_state_create_info,
            pDynamicState=None,
            layout=self.pipeline_layout,
            renderPass=self.render_pass
        )
        self.pipeline_accumulation = vk.vkCreateGraphicsPipelines(
            self.device, self.pipeline_cache, 1, [create_info], None
        )[0]

        for shader_module in shader_modules:
            vk.vkDestroyShaderModule(self.device, shader_module, None)

        # Gaussian kernel x
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "gaussian_kernel_x.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_gaussian_kernel_x = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Gaussian kernel y
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "gaussian_kernel_y.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_gaussian_kernel_y = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Integral columns
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "integral_columns.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_integral_columns = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Integral image
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "integral_image.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_integral_image = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Upper-left integral triangle
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device,
                                        os.path.join(self.shader_directory, "upper_left_integral_triangle.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_upper_left_integral_triangle = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Upper-right integral triangle
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device,
                                        os.path.join(self.shader_directory, "upper_right_integral_triangle.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_upper_right_integral_triangle = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Deformation image
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "deformation.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_deformation = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

        # Regularization
        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=create_shader_module(self.device, os.path.join(self.shader_directory, "regularization.comp.spv")),
            pName="main",
            pSpecializationInfo=None
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage, layout=self.pipeline_layout)
        self.pipeline_regularization = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache, 1, [create_info], None, None
        )[0]
        vk.vkDestroyShaderModule(self.device, create_info.stage.module, None)

    def upload_points(self, points):
        if not isinstance(points, np.ndarray):
            raise Exception("Points must be an instance of np.ndarray")

        if not points.ndim == 2:
            raise Exception("points.ndim != 2")

        if not points.shape[1] == 2:
            raise Exception("points.shape[1] != 2")

        if not points.dtype == np.float32:
            raise Exception("points.dtype != np.float32")

        # Copy points to host buffer
        points_host = Buffer.create(
            self.device, self.physical_device, points.nbytes,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )

        mapped_memory = points_host.map_memory()
        mapped_memory[:] = points.tobytes()
        points_host.unmap_memory()

        # Copy points from host to device buffer
        points_device = Buffer.create(
            self.device, self.physical_device, points.nbytes,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )

        begin_info = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        buffer_copy = vk.VkBufferCopy(0, 0, points.nbytes)
        vk.vkCmdCopyBuffer(self.command_buffer, points_host.buffer, points_device.buffer, 1, [buffer_copy])
        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[self.command_buffer])
        vk.vkQueueSubmit(self.queue, submitCount=1, pSubmits=[submit_info], fence=None)
        vk.vkQueueWaitIdle(self.queue)

        points_host.destroy()
        return points_device

    def download_points(self, points_buffer, point_count):
        points = np.zeros((point_count, 2), dtype=np.float32)

        # Copy points from device to host buffer
        points_host = Buffer.create(
            self.device, self.physical_device, points.nbytes,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )

        vk.vkResetCommandBuffer(self.command_buffer, 0)
        begin_info = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        buffer_copy = vk.VkBufferCopy(0, 0, points.nbytes)
        vk.vkCmdCopyBuffer(self.command_buffer, points_buffer.buffer, points_host.buffer, 1, [buffer_copy])
        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[self.command_buffer])
        vk.vkQueueSubmit(self.queue, submitCount=1, pSubmits=[submit_info], fence=None)
        vk.vkQueueWaitIdle(self.queue)

        # Copy points from host buffer
        mapped_memory = points_host.map_memory()
        points = np.frombuffer(mapped_memory, dtype=np.float32).reshape((point_count, 2))
        points_host.unmap_memory()

        return points

    def regularize_buffer(self, points_buffer, point_count, kernel_radius, iterations):

        # Update descriptor sets
        descriptor_buffer_infos = [
            vk.VkDescriptorBufferInfo(points_buffer.buffer, 0, points_buffer.memory_size)
        ]
        descriptor_image_infos = [
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_density.image_view, vk.VK_IMAGE_LAYOUT_GENERAL),
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_integral_columns.image_view, vk.VK_IMAGE_LAYOUT_GENERAL),
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_integral_image.image_view, vk.VK_IMAGE_LAYOUT_GENERAL),
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_upper_left_integral_triangle.image_view, vk.VK_IMAGE_LAYOUT_GENERAL),
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_upper_right_integral_triangle.image_view, vk.VK_IMAGE_LAYOUT_GENERAL),
            vk.VkDescriptorImageInfo(
                self.sampler, self.images_deformation.image_view, vk.VK_IMAGE_LAYOUT_GENERAL)
        ]
        descriptor_writes = [
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=0, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo=None, pBufferInfo=descriptor_buffer_infos[0]
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=1, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[0], pBufferInfo=None
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=2, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[1], pBufferInfo=None
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=3, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[2], pBufferInfo=None
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=4, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[3], pBufferInfo=None
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=5, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[4], pBufferInfo=None
            ),
            vk.VkWriteDescriptorSet(
                dstSet=self.descriptor_set, dstBinding=6, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptor_image_infos[5], pBufferInfo=None
            )
        ]
        vk.vkUpdateDescriptorSets(self.device, len(descriptor_writes), descriptor_writes, 0, None)

        # Prepare push constants
        push_constants = PushConstants()
        push_constants.pointCount = point_count
        push_constants.kernelRadius = kernel_radius
        push_constants_bytes = ctypes.addressof(push_constants)

        # Perform regularization
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        begin_info = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)

        for i in range(iterations):
            # Compute density image
            constant_density = point_count / (self.texture_size * self.texture_size)
            render_area = vk.VkRect2D(vk.VkOffset2D(0, 0), vk.VkExtent2D(self.texture_size, self.texture_size))
            clear_values = vk.VkClearValue(vk.VkClearColorValue([constant_density, 0.0, 0.0, 0.0]))
            render_pass_begin_info = vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.framebuffer,
                renderArea=render_area,
                clearValueCount=1,
                pClearValues=[clear_values]
            )
            vk.vkCmdBeginRenderPass(self.command_buffer, render_pass_begin_info, vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_accumulation)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       self.pipeline_layout, 0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDraw(self.command_buffer, point_count, 1, 0, 0)
            vk.vkCmdEndRenderPass(self.command_buffer)

            # Gaussian kernel x
            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                 self.pipeline_gaussian_kernel_x)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                       self.pipeline_layout, 0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdPushConstants(self.command_buffer, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, ctypes.sizeof(PushConstants), push_constants_bytes)
            vk.vkCmdDispatch(self.command_buffer, self.texture_size, 1, 1)

            # Gaussian kernel y
            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                 self.pipeline_gaussian_kernel_y)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdPushConstants(self.command_buffer, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, ctypes.sizeof(PushConstants), push_constants_bytes)
            vk.vkCmdDispatch(self.command_buffer, self.texture_size, 1, 1)

            # Compute integral columns
            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_integral_columns)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDispatch(self.command_buffer, self.texture_size, 1, 1)

            # Compute integral image
            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_integral_image)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDispatch(self.command_buffer, self.texture_size, 1, 1)

            # Compute integral triangles
            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                 self.pipeline_upper_left_integral_triangle)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDispatch(self.command_buffer, 2 * self.texture_size - 1, 1, 1)

            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                 self.pipeline_upper_right_integral_triangle)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDispatch(self.command_buffer, 2 * self.texture_size - 1, 1, 1)

            # Compute deformation
            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_deformation)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdDispatch(self.command_buffer, self.texture_size, 1, 1)

            # Perform regularization
            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

            work_group_count = (point_count + 1024 - 1) // 1024
            vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_regularization)
            vk.vkCmdBindDescriptorSets(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                       0, 1, [self.descriptor_set], 0, None)
            vk.vkCmdPushConstants(self.command_buffer, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, ctypes.sizeof(PushConstants), push_constants_bytes)
            vk.vkCmdDispatch(self.command_buffer, work_group_count, 1, 1)

            memory_barrier = vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            vk.vkCmdPipelineBarrier(
                commandBuffer=self.command_buffer,
                srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                dependencyFlags=vk.VK_DEPENDENCY_BY_REGION_BIT,
                memoryBarrierCount=1,
                pMemoryBarriers=[memory_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=None,
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=None
            )

        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer]
        )
        vk.vkQueueSubmit(self.queue, submitCount=1, pSubmits=[submit_info], fence=None)
        vk.vkQueueWaitIdle(self.queue)

    def regularize(self, points, kernel_radius, iterations):
        points_buffer = self.upload_points(points)
        self.regularize_buffer(points_buffer, points.shape[0], kernel_radius, iterations)
        return self.download_points(points_buffer, points.shape[0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    """
    The input must be a numpy.ndarray of shape (point_count x 2) with dtype of np.float32
    The points should be normalized to [-1, 1]²
    """

    COLORS = [
		np.array([217, 217, 217]) / 255.0,
		np.array([128, 177, 211]) / 255.0,
		np.array([251, 128, 114]) / 255.0,
		np.array([179, 222, 105]) / 255.0,
		np.array([253, 180,  98]) / 255.0,
		np.array([141, 211, 199]) / 255.0,
		np.array([190, 186, 218]) / 255.0,
		np.array([188, 128, 189]) / 255.0,
		np.array([255, 255, 179]) / 255.0,
		np.array([252, 205, 229]) / 255.0
	]
    
    points = []
    labels = []
    
    with open("mnist.txt", "r") as file:
        for line in file:
            x, y, label = line.split(" ")
            points.append([float(x), float(y)])
            labels.append(int(label))
    
    points = np.array(points, dtype=np.float32)
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.abs(points))

    colors = [COLORS[index] for index in labels]

    regularized = Regularizer().regularize(points, kernel_radius=16, iterations=32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(points[:, 0], points[:, 1], c=colors, s=1)
    axes[1].scatter(regularized[:, 0], regularized[:, 1], c=colors, s=1)

    for axis in axes:
        axis.set_xlim((-1.0, 1.0))
        axis.set_ylim((-1.0, 1.0))
    
    plt.tight_layout()
    plt.savefig( "figure.png" )
