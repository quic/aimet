#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
#  =============================================================================
""" The Product class is used represent the following entities in a model.
    1) a Tensor between two layers
    2) an input
    3) a constant
    4) a parameter
    """


# Forward declaration
class Op:
    """ The Op class represents a model operation. """


class Product:
    # pylint: disable=too-many-instance-attributes
    """An instance of this class represents either a tensor exchanged between a
    'producer' operation and a 'consumer' operation, or a tensor that is fed
    into a module, like an input, a constant or a parameter."""

    def __init__(self, name, shape):
        self._name = name
        self._producer = None
        self._shape = shape
        self._consumers = []

        self._is_parm = False
        self._is_model_input = False
        self._is_const = False

        self._parm_name = None
        self._impacts_in_channels = False
        self._impacts_out_channels = False
        self._impacts_groups = False

    def __repr__(self):
        """ Printable representation of the object. """
        return self._name

    @property
    def name(self):
        """ Returns the name of a Product. """
        return self._name

    @name.setter
    def name(self, new_name):
        """ Set the product name. """
        self._name = new_name

    @property
    def shape(self):
        """ Returns the shape of a Product. """
        return self._shape

    @shape.setter
    def shape(self, shape):
        """ Set the product shape. """
        self._shape = shape

    @property
    def is_parm(self):
        """ If the Product is a Parameter(weights), returns True.
        Otherwise returns False. """
        return self._is_parm

    @is_parm.setter
    def is_parm(self, tf):
        """ Sets the parameter status to True  ot False. """
        self._is_parm = tf

    @property
    def is_model_input(self):
        """ If the Product is a input to the model, returns True.
       Otherwise returns False. """
        return self._is_model_input

    @is_model_input.setter
    def is_model_input(self, tf):
        """ Sets the model input status to True  ot False. """
        self._is_model_input = tf

    @property
    def is_const(self):
        """ If the Product is a constant, returns True.
        Otherwise returns False. """
        return self._is_const

    @is_const.setter
    def is_const(self, tf):
        """ Sets the constant status to True  ot False. """
        self._is_const = tf

    @property
    def producer(self):
        """ Returns the Producer of this Product. """
        return self._producer

    @producer.setter
    def producer(self, op: Op):
        """ Sets the Producer of this Product. """
        self._producer = op

    @property
    def consumers(self):
        """ Returns the Consumers of this Product. """
        return self._consumers

    def add_consumer(self, op: Op):
        """ Adds a Consumer to this Product. """
        self._consumers.append(op)

    def set_consumers_to_null(self):
        """ Adds a Consumer to this Product. """
        self._consumers = []

    @property
    def parm_name(self):
        """ Returns the Parameter(weights) name of this Product. """
        return self._parm_name

    @parm_name.setter
    def parm_name(self, parm_name):
        """ Sets the Parameter(weights) name of this Product. """
        self._parm_name = parm_name

    @property
    def impacts_in_channels(self):
        """ If the input channels are impacted, returns True.
        Otherwise, returns False. """
        return self._impacts_in_channels

    @impacts_in_channels.setter
    def impacts_in_channels(self, tf):
        """This allows us to update a module's property 'in_channels' when one of its parameters
        (like a conv2d's weight) is reduced. Dirty solution to work around the fact that when
        applying a scenario, we reduce any product without knowledge of the (type of the) module
        that consumes the product."""
        self._impacts_in_channels = tf

    @property
    def impacts_out_channels(self):
        """ If the output channels are impacted, returns True.
        Otherwise, returns False. """
        return self._impacts_out_channels

    @impacts_out_channels.setter
    def impacts_out_channels(self, tf):
        """This allows us to update a module's property 'out_channels' when one
        of its parameters(like a conv2d's weight) is reduced. Dirty solution to
        work around the fact that when applying a scenario, we reduce any
        product without knowledge of the (type of the) module that consumes the
        product."""
        self._impacts_out_channels = tf

    @property
    def impacts_groups(self):
        """ If the groups attribute is impacted, returns True.
        Otherwise, returns False. """
        return self._impacts_groups

    @impacts_groups.setter
    def impacts_groups(self, tf):
        """This allows us to update a module's property 'out_channels' when one
        of its parameters(like a conv2d's weight) is reduced. Dirty solution to
        work around the fact that when applying a scenario, we reduce any
        product without knowledge of the (type of the) module that consumes the
        product."""
        self._impacts_groups = tf

    @property
    def numel(self):
        """ Returns number of data elements """
        num_elem = 1
        for dim in self._shape:
            num_elem *= dim
        return num_elem

    def is_inter_module(self):
        """ Returns True if the product represents an inter-module connection """
        if self._is_const or self._is_parm or self.is_model_input:
            return False
        return True
