#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/triplet_loss_with_sample_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class TripletLossWithSampleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletLossWithSampleLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    const int pair_size = 2;
    const int pair_num = blob_bottom_label_->count() / pair_size;
    for (int i = 0; i < pair_num; i++) {
        for (int j = 0; j < pair_size; j++) {
            blob_bottom_label_->mutable_cpu_data()[i*pair_size+j] = i;
        }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~TripletLossWithSampleLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossWithSampleLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossWithSampleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  //layer_param.add_loss_weight(3);
  TripletLossWithSampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(TripletLossWithSampleLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossWithSampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];

  const Dtype margin = layer_param.triplet_loss_with_sample_param().margin();
  const int pair_size = layer_param.triplet_loss_with_sample_param().pair_size();
  const int num = this->blob_bottom_label_->count();
  const int dim = this->blob_bottom_data_->count() / num;
  const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
  const Dtype* label = this->blob_bottom_label_->cpu_data();
  Dtype accum_loss = 0;
  int triplet_count = 0;
  for (int i = 0; i < num; i++) {
    int pair_id = i / pair_size;
    for (int k = pair_id*pair_size; k < (pair_id+1)*pair_size; k++) {
      if (label[i] != label[k])
        continue;
      if (k == i)
        continue;

      Dtype dist_ik = -caffe_cpu_dot(dim, bottom_data+(i*dim), bottom_data+(k*dim));
      for (int j = 0; j < num; j++) {
        if (label[j] == label[i])
          continue;

        Dtype dist_ij = -caffe_cpu_dot(dim, bottom_data+(i*dim), bottom_data+(j*dim));
        if (margin + dist_ik - dist_ij > Dtype(0.0)) {
          triplet_count += 1;
          accum_loss += margin + dist_ik - dist_ij;
        }
      }
    }    
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(triplet_count * full_loss, accum_loss, 1e-4);
}

}  // namespace caffe
