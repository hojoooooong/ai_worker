// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FFW_JOINT_STATE_BROADCASTER__VISIBILITY_CONTROL_H_
#define FFW_JOINT_STATE_BROADCASTER__VISIBILITY_CONTROL_H_

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define FFW_JOINT_STATE_BROADCASTER_EXPORT __attribute__((dllexport))
#define FFW_JOINT_STATE_BROADCASTER_IMPORT __attribute__((dllimport))
#else
#define FFW_JOINT_STATE_BROADCASTER_EXPORT __declspec(dllexport)
#define FFW_JOINT_STATE_BROADCASTER_IMPORT __declspec(dllimport)
#endif
#ifdef FFW_JOINT_STATE_BROADCASTER_BUILDING_DLL
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC FFW_JOINT_STATE_BROADCASTER_EXPORT
#else
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC FFW_JOINT_STATE_BROADCASTER_IMPORT
#endif
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC_TYPE FFW_JOINT_STATE_BROADCASTER_PUBLIC
#define FFW_JOINT_STATE_BROADCASTER_LOCAL
#else
#define FFW_JOINT_STATE_BROADCASTER_EXPORT __attribute__((visibility("default")))
#define FFW_JOINT_STATE_BROADCASTER_IMPORT
#if __GNUC__ >= 4
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC __attribute__((visibility("default")))
#define FFW_JOINT_STATE_BROADCASTER_LOCAL __attribute__((visibility("hidden")))
#else
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC
#define FFW_JOINT_STATE_BROADCASTER_LOCAL
#endif
#define FFW_JOINT_STATE_BROADCASTER_PUBLIC_TYPE
#endif

#endif  // FFW_JOINT_STATE_BROADCASTER__VISIBILITY_CONTROL_H_
