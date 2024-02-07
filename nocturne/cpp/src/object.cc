// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "object.h"

#include <algorithm>

#include "geometry/geometry_utils.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon Object::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      geometry::Vector2D(length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p1 =
      geometry::Vector2D(-length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p2 =
      geometry::Vector2D(-length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p3 =
      geometry::Vector2D(length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

void Object::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  sf::RectangleShape rect(sf::Vector2f(length_, width_));
  rect.setOrigin(length_ / 2.0f, width_ / 2.0f);
  rect.setPosition(utils::ToVector2f(position_));
  rect.setRotation(geometry::utils::Degrees(heading_));

  sf::Color col;
  if (can_block_sight_ && can_be_collided_) {
    col = color_;
  } else if (can_block_sight_ && !can_be_collided_) {
    col = sf::Color::Blue;
  } else if (!can_block_sight_ && can_be_collided_) {
    col = sf::Color::White;
  } else {
    col = sf::Color::Black;
  }

  rect.setFillColor(col);
  target.draw(rect, states);

  if (highlight_) {
    float radius = std::max(length_, width_);
    sf::CircleShape circ(radius);
    circ.setOrigin(length_ / 2.0f, width_ / 2.0f);
    circ.setPosition(utils::ToVector2f(position_));
    circ.setFillColor(sf::Color(255, 0, 0, 100));
    target.draw(circ, states);
  }

  sf::ConvexShape arrow;
  arrow.setPointCount(3);
  arrow.setPoint(0, sf::Vector2f(0.0f, -width_ / 2.0f));
  arrow.setPoint(1, sf::Vector2f(0.0f, width_ / 2.0f));
  arrow.setPoint(2, sf::Vector2f(length_ / 2.0f, 0.0f));
  arrow.setOrigin(0.0f, 0.0f);
  arrow.setPosition(utils::ToVector2f(position_));
  arrow.setRotation(geometry::utils::Degrees(heading_));
  arrow.setFillColor(sf::Color::White);
  target.draw(arrow, states);
}

void Object::InitRandomColor() {
  std::uniform_int_distribution<int32_t> dis(0, 255);
  int32_t r = dis(random_gen_);
  int32_t g = dis(random_gen_);
  int32_t b = dis(random_gen_);
  // Rescale colors to avoid dark objects.
  const int32_t max_rgb = std::max({r, g, b});
  r = r * 255 / max_rgb;
  g = g * 255 / max_rgb;
  b = b * 255 / max_rgb;
  color_ = sf::Color(r, g, b);
}

void Object::SetActionFromKeyboard() {
  // up: accelerate ; down: brake
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
    acceleration_ = 1.0f;
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
    // larger acceleration for braking than for moving backwards
    acceleration_ = velocity_.Norm() > 0 ? -2.0f : -1.0f;
  } else if (std::abs(velocity_.Norm()) < 0.05) {
    // clip to 0
    velocity_ = geometry::Vector2D(0.0f, 0.0f);
  } else {
    // friction
    acceleration_ = 0.5f * (velocity_.Norm() > 0 ? -1.0f : 1.0f);
  }

  // right: turn right; left: turn left
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
    steering_ = geometry::utils::Radians(-10.0f);
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
    steering_ = geometry::utils::Radians(10.0f);
  } else {
    steering_ = 0.0f;
  }
}

// Kinematic Bicycle Model
// https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE
void Object::KinematicBicycleStep(float dt) {
  // Forward dynamics:

    // x = trajectory.x
    // y = trajectory.y
    // vel_x = trajectory.vel_x
    // vel_y = trajectory.vel_y
    // yaw = trajectory.yaw
    // speed = jnp.sqrt(trajectory.vel_x**2 + trajectory.vel_y**2)

    // # Shape: (..., num_objects, 2)
    // action_array = self._clip_values(action.data)
    // accel, steering = jnp.split(action_array, 2, axis=-1)
    // if self._normalize_actions:
    //   accel = accel * self._max_accel
    //   steering = steering * self._max_steering
    // t = self._dt

    // new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
    // new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
    // delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
    // new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
    // new_vel = speed + accel * t
    // new_vel_x = new_vel * jnp.cos(new_yaw)
    // new_vel_y = new_vel * jnp.sin(new_yaw)
  geometry::Vector2D vel = Velocity();
  position_ = position_ + vel * dt + 0.5 * acceleration_ * geometry::Vector2D(std::cos(heading_), std::sin(heading_)) * (float)pow(dt, 2);

  heading_ = geometry::utils::AngleAdd(heading_, (float)(steering_ * (vel.Norm() * dt + 0.5 * acceleration_ * pow(dt, 2))));
  float new_vel = vel.Norm() + acceleration_ * dt;
  // TODO(ev) put back clip speed
  // Okay, so we know that the heading after the update matches perfectly! So it has to be a coordinate frome thing
  
  velocity_ = ClipSpeed(geometry::Vector2D(new_vel * cos(heading_), new_vel * sin(heading_)));
}

}  // namespace nocturne
