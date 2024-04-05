# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================
""" Function selector for quantized modules """


from typing import Any, List, Callable, Protocol, Tuple, Union, Sequence, Optional

OpArgs = Any

class _FunctionalLibrary(Protocol):
    """
    Protocol for quantized operator libraries to follow for AIMET compatibility
    """

    @staticmethod
    def get_kernel(op_key: str) -> Sequence[Tuple[Callable[[OpArgs], bool], Callable[[OpArgs], Any]]]:
        """
        Takes the kernel name as an argument and returns a sequence of (predicate, operator) pairs which take identical
        arguments. The predicate function will return True if the operator can be successfully called with the given
        inputs, False otherwise.
        """


class FunctionSelector:
    """
    Handles kernel selection for multiple operator libraries for QuantizedModules
    """

    def __init__(self,
                 functional_library: Union[List[_FunctionalLibrary], _FunctionalLibrary],
                 strict=False):
        super().__init__()
        self._functional_libraries = functional_library if isinstance(functional_library, list) else [functional_library]
        self.strict = strict

    def set_libraries(self,
                      library: Union[List[_FunctionalLibrary], _FunctionalLibrary],
                      strict: bool = False):
        """
        Set the dispatcher's operator library

        :param library: operator library or list of libraries to call into
        :param strict: If True, throw an error when no valid kernel is found
        """
        self._functional_libraries = library if isinstance(library, list) else [library]
        self.strict = strict

    def get_libraries(self) -> List[_FunctionalLibrary]:
        """
        Get the current operator libraries

        :return: the currently loaded operator library
        """
        return self._functional_libraries.copy()


    def get_impl(self, op_key: str, *args, **kwargs) -> Optional[Callable]:
        """
        Return the first function implementation with key op_key for which the predicate function returns True

        :param op_key: the key indicated which op type to retrieve
        :return: A valid implementation for the given operation if it exists
        """
        for op_lib in self._functional_libraries:
            for predicate, operator in op_lib.get_kernel(op_key):
                if predicate(*args, **kwargs):
                    return operator

        if self.strict:
            raise RuntimeError(f"No valid implementation found for op_key {op_key} with arguments "
                               f"{args}, {kwargs}")
        return None

    def is_empty(self):
        """
        Returns true if function selector has no registered library
        """
        return not bool(self._functional_libraries)
