#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OrientationLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OrientationLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_min(-3.14);
    data_filler_param.set_max(3.14);
    UniformFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-3.14);
    targets_filler_param.set_max(3.14);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~OrientationLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OrientationLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(OrientationLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    OrientationLossParameter* orientation_loss_param = layer_param.mutable_orientation_loss_param();
    bool normalize_per_positive = false; 
    orientation_loss_param->set_normalize_per_positive(normalize_per_positive); 
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    OrientationLossLayer<Dtype> layer(layer_param);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
}


}  // namespace caffe
