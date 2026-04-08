/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#ifndef LASER__LASER_HPP_
#define LASER__LASER_HPP_

#include <cmath>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


#include "ament_index_cpp/get_package_share_directory.hpp"

#include "v_marker_estimation/msg/laser_sample.hpp"
#include "v_marker_estimation/msg/laser_sample_set.hpp"
#include "v_marker_estimation/msg/laser_sample_set_list.hpp"

#include <list>
#include <set>
#include <vector>

namespace laser_processor
{
  class Sample
  {
  public:
    int   index;
    float range;
    float intensity;
    float x;
    float y;

    static Sample* Extract(int ind, const sensor_msgs::msg::LaserScan& scan);
  };

  struct CompareSample {
      bool operator() (const Sample* a, const Sample* b) const {
          return (a->index < b->index);
      }
  };


  class SampleSet : public std::set<Sample*, CompareSample>
  {
  public:
    ~SampleSet() { clear(); }

    void clear();
    geometry_msgs::msg::Point getPosition();
    std::vector<geometry_msgs::msg::Point>  getPoints();
    std::vector<v_marker_estimation::msg::LaserSample> getSamples();
  };

  class ScanProcessor
  {
    std::list<SampleSet*> clusters_;
    sensor_msgs::msg::LaserScan scan_;

  public:
    std::list<SampleSet*>& getClusters() { return clusters_; }
    std::vector<v_marker_estimation::msg::LaserSampleSet> getSampleSets();

    ScanProcessor(const sensor_msgs::msg::LaserScan& scan);
    ~ScanProcessor();

    void removeLessThan(uint32_t num);
    void splitConnected(float thresh, float intensity);
    void mergeClusters(float threshold);
  };
}
#endif //LASER__LASER_HPP_