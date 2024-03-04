// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "geometry/vector_2d.h"
#include "object.h"
#include "utils/sf_utils.h"

namespace nocturne {
#define DEFAULT_COLOR sf::Color(128, 128, 128)  // gray
#define SRC_COLOR \
  { sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Magenta }

#define MIN_TRACE_SIZE 0.25
#define MAX_TRACE_SIZE 0.9
#define TRACE_DECAY 0.1
class Vehicle : public Object {
 public:
  Vehicle() = default;

  Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& target_position, float target_heading,
          float target_speed, bool is_av, bool can_block_sight = true,
          bool can_be_collided = true, bool check_collision = true)
      : Object(id, length, width, position, heading, speed, target_position,
               target_heading, target_speed, can_block_sight, can_be_collided,
               check_collision),
        is_av_(is_av) {
    Object::InitColor(std::make_optional(DEFAULT_COLOR));
  }

  Vehicle(int64_t id, float length, float width, float max_speed,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& target_position, float target_heading,
          float target_speed, bool is_av, bool can_block_sight = true,
          bool can_be_collided = true, bool check_collision = true)
      : Object(id, length, width, max_speed, position, heading, speed,
               target_position, target_heading, target_speed, can_block_sight,
               can_be_collided, check_collision),
        is_av_(is_av) {
    Object::InitColor(std::make_optional(DEFAULT_COLOR));
  }
  void colorAsSrc(const int src_index = 0) {
    std::vector<sf::Color> src_colors = SRC_COLOR;
    Object::InitColor(src_colors[src_index]);
  }

  ObjectType Type() const override { return ObjectType::kVehicle; }

  bool is_av() const { return is_av_; }

  void makeTrace(sf::RenderTarget& target) {
    std::vector<sf::Color> src_colors = SRC_COLOR;
    std::cout << speed_ << std::endl;
    if (speed_ < 4) {  // less traces if moving slow (prevents cheetoing)
      if (trace_delay_counter < trace_delay) {
        trace_delay_counter++;
        return;
      }
      if (trace_delay_counter == trace_delay) {
        trace_delay_counter = 0;
      }
    }
    std::shared_ptr<sf::CircleShape> trace =
        std::make_shared<sf::CircleShape>(MAX_TRACE_SIZE);
    trace->setFillColor(src_colors[0]);
    float xTranform;
    float yTranform;
    // TODO : the position of the vehicle is the corner not the middle

    trace->setPosition(utils::ToVector2f(position_));
    // decay on the array of traces
    for (auto& t : traces_) {
      t->setRadius(t->getRadius() - (TRACE_DECAY));
      t->setRadius(t->getRadius() < MIN_TRACE_SIZE ? MIN_TRACE_SIZE
                                                   : t->getRadius());
    }
    traces_.push_back(trace);
  }

  const std::vector<std::shared_ptr<sf::CircleShape>>& getTraces() const {
    return traces_;
  }

 protected:
  bool is_av_;
  std::vector<std::shared_ptr<sf::CircleShape>> traces_;

 private:
  int trace_delay = 20;
  int trace_delay_counter = 0;
};

}  // namespace nocturne
