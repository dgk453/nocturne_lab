// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics/Color.hpp>

#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {
#define DEFAULT_COLOR sf::Color(128, 128, 128)  // gray
#define SRC_COLOR sf::Color::Red
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
  void colorAsSrc(
      const std::optional<sf::Color>& color = std::make_optional(SRC_COLOR)) {
    Object::InitColor(color);
    is_src_ = true;
  }

  ObjectType Type() const override { return ObjectType::kVehicle; }

  bool is_av() const { return is_av_; }

  void placeTrace() { return; }  // TODO: implement

 protected:
  bool is_av_;
  bool is_src_ = false;
};

}  // namespace nocturne
