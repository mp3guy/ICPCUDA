/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef INITIALISATION_HPP_
#define INITIALISATION_HPP_

#include <string>

/** \brief Returns number of Cuda device. */
int getCudaEnabledDeviceCount();

/** \brief Sets active device to work with. */
void setDevice(int device);

/** \brief Return devuce name for gived device. */
std::string getDeviceName(int device);

/** \brief Prints infromatoin about given cuda deivce or about all deivces
 *  \param deivce: if < 0 prints info for all devices, otherwise the function interpets is as device id.
 */
void printCudaDeviceInfo(int device = -1);

/** \brief Prints infromatoin about given cuda deivce or about all deivces
 *  \param deivce: if < 0 prints info for all devices, otherwise the function interpets is as device id.
 */
void printShortCudaDeviceInfo(int device = -1);

/** \brief Returns true if pre-Fermi generaton GPU.
  * \param device: device id to check, if < 0 checks current device.
  */
bool checkIfPreFermiGPU(int device = -1);

/** \brief Error handler. All GPU functions call this to report an error. For internal use only */
void error(const char *error_string, const char *file, const int line, const char *func = "");

#endif /* INITIALISATION_HPP_ */
