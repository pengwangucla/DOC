#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RemapVal(const int count, const Dtype *src, Dtype *dst)
{
  for(int i = 0; i < count; i ++)
  {
    Dtype input_data = src[i]; 
    if(input_data <= PI)
      dst[i] = PI/2 - input_data; 
    else if(input_data <= 2*PI)
      dst[i] = input_data-3*PI/2; 
    else
      dst[i] = 5*PI/2-input_data; 
  }
}

template <typename Dtype>
inline void SmoothL1Diff(const Dtype val, const Dtype thresh, Dtype& out)
{
    Dtype abs_val = std::abs(val);
    if (abs_val < thresh) {
      out = val;
    } else {
      out = (Dtype(0) < val) - (val < Dtype(0));
    }
}

template <typename Dtype>
void GetOrientDiff(const int count, const Dtype *src, Dtype *dst)
{
  Dtype thresh = 0.001; 
  for(int i=0; i < count; i ++)
  {
    Dtype input_data = src[i]; 
    Dtype abs_val = std::abs(input_data); 
    Dtype sign = ( abs_val <= PI || abs_val > 2*PI) ? 1 : -1; 
    SmoothL1Diff(sign*input_data, thresh, dst[i]); 
  }
}

template <typename Dtype>
void OrientationLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //CHECK_EQ(bottom[0].channel(), 1)<<"Only support single channel"; 

  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  factor_ = 1; 
  if(this->layer_param_.has_orientation_loss_param())
    factor_ = this->layer_param().orientation_loss_param().sig_factor(); 
  scale_factor_  = 1;
}

template <typename Dtype>
void OrientationLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<"Loss reshape";
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "OrientationLoss layer inputs must have the same count.";
  diff_.ReshapeLike(*bottom[0]);
  sigmoid_bottom_vec_[0] = bottom[0];
  //LOG(INFO)<<"sigmoid reshape"<<sigmoid_bottom_vec_[0]->shape().size();
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void OrientationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  Blob<Dtype> sig_input; 

  sig_input.ReshapeLike(*bottom[0]);
  bool need_normalization = (this->layer_param_.has_orientation_loss_param() &&
      this->layer_param_.orientation_loss_param().normalize_per_positive());

  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const Dtype* output_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();  

  //LOG(INFO)<<"Get diff"; 
  
  caffe_sub(count, output_data, target, this->diff_.mutable_cpu_data());

  //LOG(INFO)<<"Get abs"; 
  caffe_abs(count, this->diff_.cpu_data(), sig_input.mutable_cpu_data());

  //LOG(INFO)<<"Get offset"; 
  RemapVal(count, sig_input.cpu_data(), sig_input.mutable_cpu_data()); 

  //LOG(INFO)<<"Get rescale"<<sig_input.shape().size(); 
  caffe_scal(count, this->factor_, sig_input.mutable_cpu_data()); 

  sigmoid_bottom_vec_[0] = &sig_input;

  //LOG(INFO)<<"Sigmoid"; 
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // Stable version of loss computation from input data
  const Dtype* input_data = sig_input.cpu_data();

  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i]-log(1 + exp(input_data[i])); 
  }

  if(need_normalization){
    CHECK_EQ(bottom.size(), 3)<<"need a indicator mask for normalize_per_positive"; 
    Dtype sum = caffe_cpu_dot(count, bottom[2]->cpu_data(), bottom[2]->cpu_data()) +1;
    this->scale_factor_ =  sum / num;
    loss /= this->scale_factor_ ;
  }

  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void OrientationLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if(propagate_down.size() > 2)
    if (propagate_down[2]){
      LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to indicator inputs.";   
    }

  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype neg_one = -1.0; 

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // get the orientation diff at each pixel 
    //LOG(INFO)<<"Get Orientation Diff"; 
    GetOrientDiff(count, this->diff_.cpu_data(), this->diff_.mutable_cpu_diff()); 
    caffe_add_scalar(count, neg_one, sigmoid_output_->mutable_cpu_data());

    //LOG(INFO)<<"Multiply diff_"<<sigmoid_output_->shape().size(); 
    caffe_mul(count, sigmoid_output_->cpu_data(), this->diff_.cpu_diff(), bottom_diff); 
    // 
    
    // Scale down gradient 
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype alpha = -1*loss_weight*this->factor_/ num / this->scale_factor_; 
    caffe_scal(count, alpha, bottom_diff);

   // LOG(INFO)<<"rescale top_diff";
    //VisBlob();

  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(OrientationLossLayer, Backward);
#endif

INSTANTIATE_CLASS(OrientationLossLayer);
REGISTER_LAYER_CLASS(OrientationLoss);

}  // namespace caffe
