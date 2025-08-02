#include "SDL3/SDL_init.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

class SDLException : public std::runtime_error {
public:
  SDLException(std::string_view message)
      : std::runtime_error(std::format("{}: {}", message, SDL_GetError())) {}
};

template <typename T, void (*F)(T *)> struct PtrDeleter {
public:
  auto operator()(T *ptr) { F(ptr); }
};

template <typename T, void (*F)(T *)>
using Ptr = std::unique_ptr<T, PtrDeleter<T, F>>;
using Window = Ptr<SDL_Window, SDL_DestroyWindow>;

struct VulkanContext {
  vk::raii::Context context_;
  vk::raii::Instance instance_;
  vk::raii::SurfaceKHR surface_;
  vk::raii::PhysicalDevice physicalDevice_;
  vk::raii::Device device_;
  vk::raii::Queue queue_;

  VulkanContext(SDL_Window *window)
      : context_{makeContext()}, instance_{makeInstance(context_)},
        surface_{makeSurface(instance_, window)},
        physicalDevice_{selectPhysicalDevice(instance_, surface_)},
        device_{makeDevice(physicalDevice_)}, queue_{getQueue(device_)} {}

private:
  static auto makeContext() -> vk::raii::Context {
    auto vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        SDL_Vulkan_GetVkGetInstanceProcAddr());
    return {vkGetInstanceProcAddr};
  }

  static auto makeInstance(vk::raii::Context &context) -> vk::raii::Instance {
    Uint32 instanceRequiredExtensionCount;
    std::span instanceRequiredExtensions{
        SDL_Vulkan_GetInstanceExtensions(&instanceRequiredExtensionCount),
        instanceRequiredExtensionCount};

    return {context,
            vk::InstanceCreateInfo{}
                .setPApplicationInfo(
                    &vk::ApplicationInfo{}.setApiVersion(vk::ApiVersion14))
                .setPEnabledExtensionNames(instanceRequiredExtensions)};
  }

  static auto makeSurface(vk::raii::Instance &instance, SDL_Window *window)
      -> vk::raii::SurfaceKHR {
    VkSurfaceKHR rawSurface;
    if (!SDL_Vulkan_CreateSurface(window, *instance, nullptr, &rawSurface))
      throw SDLException("Failed to create surface");
    return {instance, rawSurface};
  }

  static auto selectPhysicalDevice(vk::raii::Instance &instance,
                                   vk::raii::SurfaceKHR &surface)
      -> vk::raii::PhysicalDevice {
    return instance.enumeratePhysicalDevices().front();
  }

  static auto makeDevice(vk::raii::PhysicalDevice &physicalDevice)
      -> vk::raii::Device {
    float queuePriorities[] = {1.0};
    std::array requiredExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::StructureChain<vk::DeviceCreateInfo,
                       vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>
        chain;

    chain.get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>()
        .setSwapchainMaintenance1(vk::True);

    return {
        physicalDevice,
        chain.get<vk::DeviceCreateInfo>()
            .setPEnabledExtensionNames(requiredExtensions)
            .setQueueCreateInfos(vk::DeviceQueueCreateInfo{}.setQueuePriorities(
                queuePriorities))};
  }

  static auto getQueue(vk::raii::Device &device) -> vk::raii::Queue {
    return {device, 0, 0};
  }
};

struct ThreadVulkanContext {
  vk::raii::CommandPool commandPool_;
  vk::raii::CommandBuffers commandBuffers_;

  ThreadVulkanContext(VulkanContext &context, size_t inFlightFramesCount)
      : commandPool_{makeCommandPool(context.device_)},
        commandBuffers_{allocateCommandBuffers(context.device_, commandPool_,
                                               inFlightFramesCount)} {}

private:
  static auto makeCommandPool(vk::raii::Device &device)
      -> vk::raii::CommandPool {
    return {device, vk::CommandPoolCreateInfo{}.setFlags(
                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer)};
  }

  static auto allocateCommandBuffers(vk::raii::Device &device,
                                     vk::raii::CommandPool &commandPool,
                                     size_t inFlightFramesCount)
      -> vk::raii::CommandBuffers {
    return {device, vk::CommandBufferAllocateInfo{}
                        .setCommandPool(commandPool)
                        .setCommandBufferCount(inFlightFramesCount)};
  }
};

struct Swapchain {
  vk::raii::SwapchainKHR swapchain_;
  std::vector<vk::Image> swapchainImages_;

  Swapchain(VulkanContext &context, SDL_Window *window)
      : swapchain_{makeSwapchain(context.device_, context.surface_,
                                 context.physicalDevice_, window)},
        swapchainImages_{swapchain_.getImages()} {}

private:
  static auto makeSwapchain(vk::raii::Device &device,
                            vk::raii::SurfaceKHR &surface,
                            vk::raii::PhysicalDevice &physicalDevice,
                            SDL_Window *window) -> vk::raii::SwapchainKHR {
    auto surfaceCapabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(surface);

    int width, height;
    if (!SDL_GetWindowSizeInPixels(window, &width, &height))
      throw SDLException("Failed to get window size");

    return {device, vk::SwapchainCreateInfoKHR{}
                        .setSurface(surface)
                        .setImageFormat(vk::Format::eR8G8B8A8Unorm)
                        .setMinImageCount(surfaceCapabilities.minImageCount + 1)
                        .setImageArrayLayers(1)
                        .setImageExtent({static_cast<uint32_t>(width),
                                         static_cast<uint32_t>(height)})
                        .setImageUsage(vk::ImageUsageFlagBits::eTransferDst)
                        .setClipped(true)};
  }
};

struct FrameSync {
  vk::raii::Semaphore imageAcquiredSemaphore_;
  vk::raii::Semaphore renderFinishedSemaphore_;
  vk::raii::Fence inFlightFence_;
  vk::raii::Fence presentFence_;

  FrameSync(VulkanContext &context)
      : imageAcquiredSemaphore_{context.device_, vk::SemaphoreCreateInfo{}},
        renderFinishedSemaphore_{context.device_, vk::SemaphoreCreateInfo{}},
        inFlightFence_{context.device_, {vk::FenceCreateFlagBits::eSignaled}},
        presentFence_{context.device_, {vk::FenceCreateFlagBits::eSignaled}} {}
};

struct FrameResources {
  std::vector<FrameSync> frames_;
  size_t currentFrameIndex_ = 0;

  FrameResources(VulkanContext &context, size_t inFlightFramesCount)
      : frames_{makeFrames(context, inFlightFramesCount)} {}

  auto AcquireNextFrame() -> FrameSync & {
    currentFrameIndex_ = (currentFrameIndex_ + 1) % frames_.size();
    return frames_.at(currentFrameIndex_);
  }

  auto GetFrameIndex() -> size_t { return currentFrameIndex_; }

private:
  static auto makeFrames(VulkanContext &context, size_t inFlightFramesCount)
      -> std::vector<FrameSync> {
    std::vector<FrameSync> frames;
    frames.reserve(inFlightFramesCount);
    for (auto _ : std::views::iota(0u, inFlightFramesCount))
      frames.emplace_back(context);
    return frames;
  }
};

const size_t IN_FLIGHT_FRAMES_COUNT = 2;

class App {
public:
  App() {}

  auto Init(std::span<char *> args) -> SDL_AppResult {
    if (!SDL_Init(SDL_INIT_VIDEO))
      throw SDLException("Failed to init SDL");

    if (!SDL_Vulkan_LoadLibrary(nullptr))
      throw SDLException("Failed to load vulkan library");

    window_.reset(SDL_CreateWindow("Codotaku", 800, 600,
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN |
                                       SDL_WINDOW_HIDDEN));

    if (!window_)
      throw SDLException("Failed to create window");

    vulkanContext.emplace(window_.get());
    swapchain_.emplace(*vulkanContext, window_.get());
    threadVulkanContext.emplace(*vulkanContext, IN_FLIGHT_FRAMES_COUNT);
    frameResources_.emplace(*vulkanContext, IN_FLIGHT_FRAMES_COUNT);

    SDL_ShowWindow(window_.get());

    return SDL_APP_CONTINUE;
  }

  auto Iterate() -> SDL_AppResult {
    auto &currentFrame = frameResources_->AcquireNextFrame();
    (void)vulkanContext->device_.waitForFences(
        {*currentFrame.inFlightFence_, *currentFrame.presentFence_}, true,
        std::numeric_limits<Uint64>().max());

    auto [result, imageIndex] = swapchain_->swapchain_.acquireNextImage(
        std::numeric_limits<Uint64>().max(),
        currentFrame.imageAcquiredSemaphore_);

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      swapchain_ = Swapchain{*vulkanContext, window_.get()};
      return SDL_APP_CONTINUE;
    }

    vulkanContext->device_.resetFences(
        {*currentFrame.inFlightFence_, *currentFrame.presentFence_});

    auto &commandBuffer =
        threadVulkanContext->commandBuffers_[frameResources_->GetFrameIndex()];
    auto &image = swapchain_->swapchainImages_.at(imageIndex);

    commandBuffer.reset();
    commandBuffer.begin(vk::CommandBufferBeginInfo{}.setFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{}
            .setImage(image)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                    .setLayerCount(1)
                    .setLevelCount(1)
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)));
    commandBuffer.clearColorImage(
        image, vk::ImageLayout::eTransferDstOptimal,
        vk::ClearColorValue{}.setFloat32({1.0, 0.0, 0.0, 1.0}),
        vk::ImageSubresourceRange{}
            .setLayerCount(1)
            .setLevelCount(1)
            .setAspectMask(vk::ImageAspectFlagBits::eColor));
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{}
            .setImage(image)
            .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setSubresourceRange(
                vk::ImageSubresourceRange{}
                    .setLayerCount(1)
                    .setLevelCount(1)
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)));
    commandBuffer.end();
    vk::PipelineStageFlags waitDstStageMask =
        vk::PipelineStageFlagBits::eTransfer;
    vulkanContext->queue_.submit(
        vk::SubmitInfo{}
            .setCommandBuffers(*commandBuffer)
            .setWaitSemaphores(*currentFrame.imageAcquiredSemaphore_)
            .setSignalSemaphores(*currentFrame.renderFinishedSemaphore_)
            .setPWaitDstStageMask(&waitDstStageMask),
        currentFrame.inFlightFence_);

    vk::StructureChain<vk::PresentInfoKHR, vk::SwapchainPresentFenceInfoEXT>
        presentChain;

    presentChain.get<vk::SwapchainPresentFenceInfoEXT>().setFences(
        *currentFrame.presentFence_);

    result = vulkanContext->queue_.presentKHR(
        presentChain.get<vk::PresentInfoKHR>()
            .setSwapchains(*swapchain_->swapchain_)
            .setImageIndices(imageIndex)
            .setWaitSemaphores(*currentFrame.renderFinishedSemaphore_));

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
  }

  auto Event(SDL_Event *event) -> SDL_AppResult {
    switch (event->type) {
    case SDL_EVENT_QUIT:
      return SDL_APP_SUCCESS;
    default:
      return SDL_APP_CONTINUE;
    }
  }

  auto Quit(SDL_AppResult result) -> void { vulkanContext->device_.waitIdle(); }

private:
  Window window_;

  std::optional<VulkanContext> vulkanContext;
  std::optional<ThreadVulkanContext> threadVulkanContext;
  std::optional<Swapchain> swapchain_;
  std::optional<FrameResources> frameResources_;
};

SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {
  auto app = new App;
  *appstate = app;
  return app->Init({argv, static_cast<size_t>(argc)});
}

SDL_AppResult SDL_AppIterate(void *appstate) {
  auto app = static_cast<App *>(appstate);
  return app->Iterate();
}

SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
  auto app = static_cast<App *>(appstate);
  return app->Event(event);
}

void SDL_AppQuit(void *appstate, SDL_AppResult result) {
  auto app = static_cast<App *>(appstate);
  app->Quit(result);
  delete app;
}