#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/l2n_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_ = bottom[0]->num();
  int channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();

  top[0]->Reshape(num_, channels_, height_, width_);
  sum_multiplier_.Reshape(1, channels_, 1, 1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i)
      multiplier_data[i] = 1.;
  square_.Reshape(num_, channels_, height_, width_);
  norm_.Reshape(num_, 1, height_, width_);
  temp_dot_.Reshape(num_, 1, height_, width_);
}

template <typename Dtype>
void L2NLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* square_data = square_.mutable_cpu_data();
    Dtype* norm_data = norm_.mutable_cpu_data();
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int dim = bottom[0]->count() / bottom[0]->num();
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    caffe_copy(bottom[0]->count(), bottom_data, square_data);
    // do the normalization.
    for (int i = 0; i < num; ++i) {
        // square each element
        caffe_sqr<Dtype>(dim, square_data + i * dim, square_data + i * dim);
        // sum acorss the channel
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1.,
                square_data + i * dim, sum_multiplier_.cpu_data(), 0.,
                norm_data + i * spatial_dim);
        // root the square norm_data
        caffe_powx<Dtype>(spatial_dim, norm_data + i * spatial_dim, 0.5,
                norm_data + i * spatial_dim);
        // division
        for (int j = 0; j < channels; ++j) {
            caffe_div(spatial_dim, top_data + top[0]->offset(i, j),
                norm_data + i * spatial_dim, top_data + top[0]->offset(i,j));
        }
    }
}

template <typename Dtype>
void L2NLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* norm_data = norm_.mutable_cpu_data();
    Dtype* temp_dot_data = temp_dot_.mutable_cpu_data();
    Dtype* temp_data = square_.mutable_cpu_data(); // just reuse the square_
    int num = top[0]->num();
    int channels = top[0]->channels();
    int dim = top[0]->count() / top[0]->num();
    int spatial_dim = top[0]->height() * top[0]->width();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
    for (int i = 0; i < num; ++i) {
        // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
        // compute dot(top_diff, top_data)
        for (int k = 0; k < spatial_dim; ++k) {
            temp_dot_data[i*spatial_dim + k] = caffe_cpu_strided_dot<Dtype>(channels,
                    top_diff + i * dim + k, spatial_dim,
                    top_data + i * dim + k, spatial_dim) /
                (norm_data[i*spatial_dim + k] * norm_data[i*spatial_dim + k]);
        }
        // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
        for (int j = 0; j < channels; ++j) {
            caffe_div(spatial_dim, bottom_diff + top[0]->offset(i, j),
                norm_data + i*spatial_dim, bottom_diff + top[0]->offset(i, j));
            caffe_mul(spatial_dim, bottom_data + top[0]->offset(i, j),
                temp_dot_data + i*spatial_dim, temp_data + top[0]->offset(i, j));
            caffe_axpy(spatial_dim, Dtype(-1.0),temp_data + top[0]->offset(i ,j),
                bottom_diff + top[0]->offset(i, j));
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(L2NLayer);
#endif

INSTANTIATE_CLASS(L2NLayer);
REGISTER_LAYER_CLASS(L2N);

}  // namespace caffe
