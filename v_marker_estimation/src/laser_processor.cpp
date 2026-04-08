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


#include <v_marker_estimation/laser_processor.hpp>


namespace laser_processor
{

Sample* Sample::Extract(int ind, const sensor_msgs::msg::LaserScan& scan)
{
  Sample* s = new Sample();

  s->index = ind;
  s->range = scan.ranges[ind];
  s->x = std::cos(scan.angle_min + ind * scan.angle_increment) * s->range;
  s->y = std::sin(scan.angle_min + ind * scan.angle_increment) * s->range;
  s->intensity = (!scan.intensities.empty()) ? scan.intensities[ind] : 0.0f;

  if (s->range > scan.range_min && s->range < scan.range_max)
  {
    return s;
  }
  else
  {
    delete s;
    return nullptr;
  }
}

void SampleSet::clear()
{
  for (iterator i = begin(); i != end(); ++i)
    delete (*i);
  std::set<Sample*, CompareSample>::clear();
}

geometry_msgs::msg::Point SampleSet::getPosition()
{
  float x_mean = 0.0;
  float y_mean = 0.0;
  for (iterator i = begin(); i != end(); ++i)
  {
    x_mean += ((*i)->x) / size();
    y_mean += ((*i)->y) / size();
  }

  geometry_msgs::msg::Point point;
  point.x = x_mean;
  point.y = y_mean;
  point.z = 0.0;

  return point;
}

std::vector<geometry_msgs::msg::Point> SampleSet::getPoints()
{
  std::vector<geometry_msgs::msg::Point> points;
  for (iterator i = begin(); i != end(); ++i)
  {
    geometry_msgs::msg::Point p;
    p.x = ((*i)->x);
    p.y = ((*i)->y);
    points.push_back(p);
  }

  return points;
}

std::vector<v_marker_estimation::msg::LaserSample> SampleSet::getSamples()
{
  std::vector<v_marker_estimation::msg::LaserSample> samples;
  for (iterator i = begin(); i != end(); ++i)
  {
    v_marker_estimation::msg::LaserSample p;
    p.index = ((*i)->index);
    p.range = ((*i)->range);
    p.intensity = ((*i)->intensity);
    p.x = ((*i)->x);
    p.y = ((*i)->y);
    samples.push_back(p);
  }

  return samples;
}

ScanProcessor::ScanProcessor(const sensor_msgs::msg::LaserScan& scan)
{
  scan_ = scan;

  SampleSet* cluster = new SampleSet;

  for (size_t i = 0; i < scan.ranges.size(); i++)
  {
    Sample* s = Sample::Extract(i, scan);

    if (s != nullptr)
      cluster->insert(s);
  }

  clusters_.push_back(cluster);
}

ScanProcessor::~ScanProcessor()
{
  for (auto c : clusters_)
    delete c;
}

std::vector<v_marker_estimation::msg::LaserSampleSet> ScanProcessor::getSampleSets()
{
  std::vector<v_marker_estimation::msg::LaserSampleSet> samplesets;
  int i = 1;
  for (auto c_iter = clusters_.begin(); c_iter != clusters_.end(); ++c_iter)
  {
    v_marker_estimation::msg::LaserSampleSet sampleset;
    sampleset.cluster_id = i++;
    sampleset.sample_set = (*c_iter)->getSamples();

    geometry_msgs::msg::Point p = (*c_iter)->getPosition();
    geometry_msgs::msg::Point mean_p;
    mean_p.x = p.x;
    mean_p.y = p.y;
    sampleset.mean_p = mean_p;
    samplesets.push_back(sampleset);
  }

  return samplesets;
}

void ScanProcessor::removeLessThan(uint32_t num)
{
  for (auto c_iter = clusters_.begin(); c_iter != clusters_.end();)
  {
    if ((*c_iter)->size() < num)
    {
      delete (*c_iter);
      c_iter = clusters_.erase(c_iter);
    }
    else
    {
      ++c_iter;
    }
  }
}

void ScanProcessor::splitConnected(float thresh,float intensity)
{
  std::list<SampleSet*> tmp_clusters;

  for (auto c_iter = clusters_.begin(); c_iter != clusters_.end(); ++c_iter)
  {
    while ((*c_iter)->size() > 0)
    {
      SampleSet::iterator s_first = (*c_iter)->begin();
      std::list<Sample*> sample_queue;
      sample_queue.push_back(*s_first);
      (*c_iter)->erase(s_first);
      for (auto s_q = sample_queue.begin(); s_q != sample_queue.end(); ++s_q)
      {
        int expand = static_cast<int>(std::asin(thresh / (*s_q)->range) / scan_.angle_increment);

        SampleSet::iterator s_rest = (*c_iter)->begin();
        while (s_rest != (*c_iter)->end() && (*s_rest)->index < (*s_q)->index + expand)
        {
          if (std::sqrt(std::pow((*s_q)->x - (*s_rest)->x, 2.0f) + std::pow((*s_q)->y - (*s_rest)->y, 2.0f)) < thresh &&
           (*s_q)->intensity >= intensity)
          {
            sample_queue.push_back(*s_rest);
            (*c_iter)->erase(s_rest++);
          }
          else
          {
            ++s_rest;
          }
        }
      }

      SampleSet* c = new SampleSet;
      for (auto s_q = sample_queue.begin(); s_q != sample_queue.end(); ++s_q)
        c->insert(*s_q);

      tmp_clusters.push_back(c);
    }

    delete (*c_iter);
  }

  clusters_.clear();
  clusters_.insert(clusters_.begin(), tmp_clusters.begin(), tmp_clusters.end());
}

void ScanProcessor::mergeClusters(float threshold) {
    if (clusters_.size() < 2) {
        return;
    }

    auto first_cluster = clusters_.front();
    auto last_cluster = clusters_.back();

    double first_cluster_x = (*first_cluster->begin())->x;
    double first_cluster_y = (*first_cluster->begin())->y;
    double last_cluster_x = (*last_cluster->rbegin())->x;
    double last_cluster_y = (*last_cluster->rbegin())->y;

    double diff_x = std::abs(first_cluster_x - last_cluster_x);
    double diff_y = std::abs(first_cluster_y - last_cluster_y);
    double distance = std::sqrt(diff_x * diff_x + diff_y * diff_y);
    
    if (distance < threshold) {
        first_cluster->insert(last_cluster->begin(), last_cluster->end());
        clusters_.pop_back();
    }
}

}; // namespace laser_processor
