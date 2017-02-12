#ifndef CAFFE_TRIPLET_LOSS_WITH_SAMPLE_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_WITH_SAMPLE_LAYER_HPP_ 
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	class TripletLossWithSampleLayer : public LossLayer<Dtype> {
	public:
		explicit TripletLossWithSampleLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), iter_count_(0) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "TripletLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int iter_count_;
		void select_triplet(const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> dist_;
		Blob<Dtype> flag_;
	};
}

#endif // CAFFE_TRIPLET_LOSS_WITH_SAMPLE_LAYER_HPP_
