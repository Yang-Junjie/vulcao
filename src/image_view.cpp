#include "vulcao/image_view.h"

#include "vulcao/image.h"

#include <stdexcept>
#include <utility>

namespace vulcao {
namespace {

vk::ImageViewType default_view_type(const Image& image) {
    switch (image.image_type()) {
        case vk::ImageType::e1D:
            return image.array_layers() > 1 ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
        case vk::ImageType::e3D:
            return vk::ImageViewType::e3D;
        default:
            if (image.cube_compatible())
                return image.array_layers() > 6 ? vk::ImageViewType::eCubeArray
                                                : vk::ImageViewType::eCube;
            return image.array_layers() > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    }
}

}

ImageView::~ImageView() {
    destroy();
}

ImageView::ImageView(ImageView&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      view_(std::exchange(other.view_, vk::ImageView{})),
      format_(std::exchange(other.format_, vk::Format::eUndefined)),
      range_(std::exchange(other.range_, vk::ImageSubresourceRange{})) {}

ImageView& ImageView::operator=(ImageView&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        view_ = std::exchange(other.view_, vk::ImageView{});
        format_ = std::exchange(other.format_, vk::Format::eUndefined);
        range_ = std::exchange(other.range_, vk::ImageSubresourceRange{});
    }
    return *this;
}

ImageView ImageView::create(vk::Device device, const vk::ImageViewCreateInfo& info) {
    if (!info.image)
        throw std::runtime_error("ImageView::create: invalid image");

    ImageView view;
    view.device_ = device;
    view.view_ = device.createImageView(info);
    view.format_ = info.format;
    view.range_ = info.subresourceRange;
    return view;
}

ImageView ImageView::create(vk::Device device, const Image& image) {
    return create(device, image, default_view_type(image), image.format());
}

ImageView ImageView::create(vk::Device device,
                            const Image& image,
                            vk::ImageViewType type,
                            vk::Format format,
                            vk::ImageSubresourceRange range) {
    if (!image.valid())
        throw std::runtime_error("ImageView::create: invalid image");

    if (format == vk::Format::eUndefined)
        format = image.format();

    if (range.levelCount == 0 || range.layerCount == 0)
        range = image.subresource_range();

    if (range.aspectMask == vk::ImageAspectFlags{})
        range.aspectMask = image_aspect_for_format(format);

    return create(device,
                  vk::ImageViewCreateInfo{
                      .image = image.handle(),
                      .viewType = type,
                      .format = format,
                      .subresourceRange = range,
                  });
}

void ImageView::destroy() {
    if (view_)
        device_.destroyImageView(view_);

    device_ = nullptr;
    view_ = nullptr;
    format_ = vk::Format::eUndefined;
    range_ = vk::ImageSubresourceRange{};
}

}
