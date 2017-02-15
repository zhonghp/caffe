#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_loss_with_sample_layer.hpp"

using namespace std;

namespace caffe {

	int random_func(int i) {
		return caffe_rng_rand() % i;
	}

	template <typename Dtype>
	void TripletLossWithSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

		LossLayer<Dtype>::Reshape(bottom, top);

		TripletLossWithSampleParameter loss_param = this->layer_param_.triplet_loss_with_sample_param();
		int pair_size = loss_param.pair_size();
		int pair_num = bottom[0]->num() / pair_size;

		dist_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
		flag_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
		cluster_dist_.Reshape(pair_num, bottom[0]->num(), 1, 1);
		cluster_feat_.Reshape(pair_num, bottom[0]->channels(), 1, 1);
	}

	template <typename Dtype>
	void TripletLossWithSampleLayer<Dtype>::select_triplet(const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		int count = bottom[0]->count();
		int num = bottom[0]->num();
		int dim = count / num;

		TripletLossWithSampleParameter loss_param = this->layer_param_.triplet_loss_with_sample_param();
		int num_negative = loss_param.num_negative();
		Dtype random_ratio = loss_param.random_ratio();
		Dtype hard_ratio = loss_param.hard_ratio();
		Dtype margin = loss_param.margin();
		int pair_size = loss_param.pair_size();
		bool cluster = loss_param.cluster();

		int pair_num = num / pair_size;
		int hard_num = num_negative * hard_ratio;
		int random_num = num_negative * random_ratio;

		Dtype* dist_data = this->dist_.mutable_cpu_data();
		Dtype* flag_data = this->flag_.mutable_cpu_data();
		Dtype* cluster_dist_data = this->cluster_dist_.mutable_cpu_data();
		for (int i = 0; i < num * num; i++) {
			dist_data[i] = 0;
			flag_data[i] = 0;
		}
		for (int i = 0; i < pair_num * num; i++) {
			cluster_dist_data[i] = 0;
		}

		// calculate the distance
		for (int i = 0; i < num; i++) {
			for (int j = i+1; j < num; j++) {
				Dtype dist = -caffe_cpu_dot(dim, bottom_data + (i*dim), bottom_data + (j*dim));
				dist_data[i*num + j] = dist;
				dist_data[j*num + i] = dist;
			}

			//calculate the cluster distance
			if ((i+1) % pair_size == 0) {
				int pair_id = (i+1) / pair_size - 1;
				for (int j = 0; j < num; j++) {
					Dtype dist = 0;
					for (int k = 0; k < pair_size; k++)
						dist += dist_data[(i-k)*num + j];
					dist /= pair_size;
					cluster_dist_data[pair_id*num + j] = dist;
				}
			}
		}

		vector<int> hard_neg_ids;
		vector<int> random_neg_ids;
		vector< pair<Dtype, int> > neg_pairs;
		for (int i = 0; i < num; i += pair_size) {
			neg_pairs.clear();
			hard_neg_ids.clear();
			random_neg_ids.clear();

			int pair_id = i / pair_size;
			for (int k = 0; k < pair_size; k++) {
				if (label[i+k] != label[i])
					continue;

				// do not consider <a, a, n> triplets
				if (k == 0 && cluster == false)
					continue;

				for (int j = 0; j < num; j++) {
					if (label[j] == label[i])
						continue;

					if (cluster) {
						Dtype dist = margin + cluster_dist_data[pair_id*num + i+k] - cluster_dist_data[pair_id*num + j];
						if (dist > Dtype(0.0))
							neg_pairs.push_back(make_pair(cluster_dist_data[pair_id*num + j], j));
					} else {
						Dtype dist = margin + dist_data[i*num + i+k] - dist_data[i*num + j];
						if (dist > Dtype(0.0))
							neg_pairs.push_back(make_pair(dist_data[i*num + j], j));
					}
				}

				//when num_negative equals to 0, we should consider all the negative samples.
				if (num_negative == 0 || neg_pairs.size() <= num_negative) {
					for (int j = 0; j < neg_pairs.size(); j++) {
						int id = neg_pairs[j].second;

						//the negative sample j is selected by the positive pair <i, i+k>
						flag_data[(i+k)*num + id] = 1;
					}

					continue;
				}

				sort(neg_pairs.begin(), neg_pairs.end());
				for (int j = 0; j < num_negative; j++)
					hard_neg_ids.push_back(neg_pairs[j].second);
				for (int j = num_negative; j < neg_pairs.size(); j++)
					random_neg_ids.push_back(neg_pairs[j].second);

				std::random_shuffle(hard_neg_ids.begin(), hard_neg_ids.end(), random_func);
				for (int j = 0; j < min(hard_num, (int)hard_neg_ids.size()); j++)
					flag_data[(i+k)*num + hard_neg_ids[j]] = 1;

				for (int j = hard_num; j < hard_neg_ids.size(); j++)
					random_neg_ids.push_back(hard_neg_ids[j]);
				std::random_shuffle(random_neg_ids.begin(), random_neg_ids.end(), random_func);
				for (int j = 0; j < min(random_num, (int)random_neg_ids.size()); j++)
					flag_data[(i+k)*num + random_neg_ids[j]] = 1;
			}
		}
	}

	template <typename Dtype>
	void TripletLossWithSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

		int num = bottom[0]->num();

		const Dtype* dist_data = this->dist_.cpu_data();
		const Dtype* flag_data = this->flag_.cpu_data();

		TripletLossWithSampleParameter loss_param = this->layer_param_.triplet_loss_with_sample_param();
		Dtype margin = loss_param.margin();
		int pair_size = loss_param.pair_size();
		bool mirror = loss_param.mirror();
		bool cluster = loss_param.cluster();

		select_triplet(bottom);

		Dtype loss(0.0);
		Dtype dist(0.0);
		int triplet_count = 0;
		const Dtype* cluster_dist_data = this->cluster_dist_.cpu_data();
		for (int i = 0; i < num; i += pair_size) {
			int pair_id = i / pair_size;
			for (int k = 0; k < pair_size; k++) {
				// do not consider <a, a, n> triplets
				if (k == 0 && cluster == false)
					continue;

				for (int j = 0; j < num; j++) {
					//the negative sample j is selected by the positive pair <i, i+k>
					if (flag_data[(i+k)*num + j] == 0)
						continue;

					if (cluster) {
						dist = margin + cluster_dist_data[pair_id*num + i+k] - cluster_dist_data[pair_id*num + j];
						if (dist > Dtype(0.0)) {
							loss += dist;
							triplet_count += 1;
						}
					} else {
						//<a, p, n>
						dist = margin + dist_data[i*num + i+k] - dist_data[i*num + j];
						if (dist > Dtype(0.0)) {
							loss += dist;
							triplet_count += 1;
						}

						//<p, a, n>
						if (mirror) {
							dist = margin + dist_data[i*num + i+k] - dist_data[(i+k)*num + j];
							if (dist > Dtype(0.0)) {
								loss += dist;
								triplet_count += 1;
							}
						}
					}
				}
			}
		}

		if (triplet_count != 0)
			loss = loss / triplet_count;
		top[0]->mutable_cpu_data()[0] = loss;

		const int magic_number = 2500;
		if (iter_count_ % magic_number == 0)
			LOG(INFO) << "Totally " << triplet_count << " <a, p, n> triplets.";

		iter_count_ += 1;
	}

	template <typename Dtype>
	void TripletLossWithSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		int count = bottom[0]->count();
		int num = bottom[0]->num();
		int dim = count / num;

		const Dtype* dist_data = this->dist_.cpu_data();
		const Dtype* flag_data = this->flag_.cpu_data();

		TripletLossWithSampleParameter loss_param = this->layer_param_.triplet_loss_with_sample_param();
		Dtype margin = loss_param.margin();
		int pair_size = loss_param.pair_size();
		bool mirror = loss_param.mirror();
		bool cluster = loss_param.cluster();

		int pair_num = num / pair_size;
		Dtype* cluster_feat_data = this->cluster_feat_.mutable_cpu_data();
		Dtype* cluster_feat_diff = this->cluster_feat_.mutable_cpu_diff();
		for (int i = 0; i < pair_num * dim; i++) {
			cluster_feat_data[i] = 0;
			cluster_feat_diff[i] = 0;
		}

		for (int i = 0; i < count; i++)
			bottom_diff[i] = 0;

		Dtype dist(0.0);
		int triplet_count = 0;
		const Dtype* cluster_dist_data = this->cluster_dist_.cpu_data();
		for (int i = 0; i < num; i+= pair_size) {
			const Dtype* anc_feat = bottom_data + i*dim;
			Dtype* anc_diff = bottom_diff + i*dim;

			int pair_id = i / pair_size;
			for (int k = 0; k < pair_size; k++) {
				const Dtype* feat = bottom_data + (i+k)*dim;
				for (int d = 0; d < dim; d++) {
					cluster_feat_data[pair_id*dim + d] += feat[d] / pair_size;
				}
			}

			for (int k = 0; k < pair_size; k++) {
				// do not consider <a, a, n> triplets
				if (k == 0 && cluster == false)
					continue;

				const Dtype* pos_feat = bottom_data + (i+k)*dim;
				Dtype* pos_diff = bottom_diff + (i+k)*dim;

				for (int j = 0; j < num; j++) {
					//the negative sample j is selected by the positive pair <i, i+k>
					if (flag_data[(i+k)*num + j] == 0)
						continue;

					const Dtype* neg_feat = bottom_data + j*dim;
					Dtype* neg_diff = bottom_diff + j*dim;

					if (cluster) {
						dist = margin + cluster_dist_data[pair_id*num + i+k] - cluster_dist_data[pair_id*num + j];
						if (dist > Dtype(0.0)) {
							triplet_count += 1;
							for (int d = 0; d < dim; d++) {
								pos_diff[d] += -cluster_feat_data[pair_id*dim + d];
								neg_diff[d] += cluster_feat_data[pair_id*dim + d];
								cluster_feat_diff[pair_id*dim + d] += (neg_feat[d] - pos_feat[d]);
							}
						}
					} else {
						//<a, p, n>
						dist = margin + dist_data[i*num + i+k] - dist_data[i*num + j];
						if (dist > Dtype(0.0)) {
							triplet_count += 1;
							for (int d = 0; d < dim; d++) {
								anc_diff[d] += (neg_feat[d] - pos_feat[d]);
								pos_diff[d] += -anc_feat[d];
								neg_diff[d] += anc_feat[d];
							}
						}

						//<p, a, n>
						if (mirror) {
							dist = margin + dist_data[i*num + i+k] - dist_data[(i+k)*num + j];
							if (dist > Dtype(0.0)) {
								triplet_count += 1;
								for (int d = 0; d < dim; d++) {
									anc_diff[d] += -pos_feat[d];
									pos_diff[d] += neg_feat[d] - anc_feat[d];
									neg_diff[d] += pos_feat[d];
								}
							}
						}
					}
				}
			}

			// do not propagate gradient to each positive samples
//			for (int k = 0; k < pair_size; k++) {
//				Dtype* pos_diff = bottom_diff + (i+k)*dim;
//				for (int d = 0; d < dim; d++) {
//					pos_diff[d] += cluster_feat_diff[pair_id*dim + d] / pair_size;
//				}
//			}
		}

	    const Dtype loss_weight = top[0]->cpu_diff()[0];
		for (int i = 0; i < count; i++)
			if (triplet_count != 0)
				bottom_diff[i] = bottom_diff[i] * loss_weight / triplet_count;
	}

#ifdef CPU_ONLY
	STUB_GPU(TripletLossWithSampleLayer);
#endif

	INSTANTIATE_CLASS(TripletLossWithSampleLayer);
	REGISTER_LAYER_CLASS(TripletLossWithSample);
}
