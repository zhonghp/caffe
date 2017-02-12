#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/l2n_layer.hpp"

#include "thrust/device_vector.h"

namespace caffe {

template <typename Dtype>
    __global__ void kernel_channel_sum(const int num, const int channels,
                const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
      CUDA_KERNEL_LOOP(index, num * spatial_dim) {
          int n = index / spatial_dim;
          int s = index % spatial_dim;
          Dtype sum = 0;
          for (int c = 0; c < channels; ++c) {
                sum += data[(n * channels + c) * spatial_dim + s];
              }
          channel_sum[index] = sum;
        }
    }

template <typename Dtype>
    __global__ void kernel_channel_mul(const int num, const int channels,
                const int spatial_dim, Dtype* data, const Dtype* channel_sum) {
      CUDA_KERNEL_LOOP(index, num * spatial_dim) {
          int n = index / spatial_dim;
          int s = index % spatial_dim;
          for (int c = 0; c < channels; ++c) {
                data[(n * channels + c) * spatial_dim + s] *= channel_sum[index];
              }
        }
    }

template <typename Dtype>
    __global__ void kernel_channel_div(const int num, const int channels,
                const int spatial_dim, Dtype* data, const Dtype* channel_sum) {
      CUDA_KERNEL_LOOP(index, num * spatial_dim) {
          int n = index / spatial_dim;
          int s = index % spatial_dim;
          for (int c = 0; c < channels; ++c) {
                data[(n * channels + c) * spatial_dim + s] /= channel_sum[index];
              }
        }
    }

template <typename Dtype>
    __global__ void kernel_channel_dot(const int num, const int channels,
                const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
                    Dtype* channel_dot) {
      CUDA_KERNEL_LOOP(index, num * spatial_dim) {
          int n = index / spatial_dim;
          int s = index % spatial_dim;
          Dtype dot = 0;
          for (int c = 0; c < channels; ++c) {
                dot += (data_1[(n * channels + c) * spatial_dim + s]
                                  * data_2[(n * channels + c) * spatial_dim + s]);
              }
          channel_dot[index] = dot;
        }
    }

template <typename Dtype>
    void L2NLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
              const  vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      Dtype* square_data = square_.mutable_gpu_data();
      Dtype* norm_data = norm_.mutable_gpu_data();
      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      int spatial_dim = bottom[0]->height() * bottom[0]->width();
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
      caffe_copy(bottom[0]->count(), bottom_data, square_data);
    
      // square
      caffe_gpu_powx<Dtype>(bottom[0]->count(), square_data, Dtype(2.0), square_data);
      //sum cross channel
      kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, square_data,
                        norm_data);
      // square root
      caffe_gpu_powx<Dtype>(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
      // divide
      kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
                        norm_data);
    }

template <typename Dtype>
    void L2NLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* top_data = top[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* norm_data = norm_.mutable_gpu_data();
      Dtype* temp_dot_data = temp_dot_.mutable_gpu_data();
      Dtype* temp_data = square_.mutable_gpu_data();//just reuse the square_
      int num = top[0]->num();
      int channels = top[0]->channels();
      int spatial_dim = top[0]->height() * top[0]->width();
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
      caffe_copy(top[0]->count(), bottom_data, temp_data);
    
      // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
      // temp_dot_data = dot(t_diff, t_data)
      kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
                        temp_dot_data);
      // temp_dot_data /= (norm)^2
      caffe_gpu_div<Dtype>(num * spatial_dim, temp_dot_data, norm_data, temp_dot_data);
      caffe_gpu_div<Dtype>(num * spatial_dim, temp_dot_data, norm_data, temp_dot_data);
      // bottom_diff = top_diff, bottom_diff /= norm
      kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
                        norm_data);
      // temp_data = bottom_data, temp_data *= temp_dot_data
      kernel_channel_mul<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, temp_data,
                        temp_dot_data); 
      // bottom_diff += -temp_data
      caffe_gpu_axpy<Dtype>(top[0]->count(), Dtype(-1.0), temp_data, 
                    bottom_diff);
    }

//INSTANTIATE_CLASS(L2NLayer);

INSTANTIATE_LAYER_GPU_FUNCS(L2NLayer);

}  // namespace caffe
