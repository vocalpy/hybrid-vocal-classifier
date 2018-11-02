#!/usr/bin/env python

#### Copyright (c) Clovis Fabricio Costa
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# this version is from https://github.com/sjparkinson/random-passwd


"""Alternate random number generator using random.org http
service as source.

RANDOM.ORG is a true random number service that generates randomness
via atmospheric noise.

To use just create a instance of the `RandomDotOrg` class, and use it as
you would use a `random.Random` instance.
"""

__version__ = '0.1.3a1'
__url__ = 'http://pypi.python.org/pypi/randomdotorg'
__all__ = ['RandomDotOrg']
__author__ = "Clovis Fabricio <nosklo at gmail dot com>"
__license__ = "GPL-3"


import random
import urllib.request, urllib.parse, urllib.error

    
def _fetch_randomorg(service, **kwargs):
    """Internal function to make a fetch in a random.org service.
    >>> _fetch_randomorg('numbers', num=3, min=10, max=20)
    ['15', '11', '18']
    """
    url = "https://www.random.org/%s/?%s"
    options = dict(format='plain', num=1, col=1, min=0, base=10) # default options
    options.update(kwargs)
    url = url % (service, urllib.parse.urlencode(options))
    headers = {'User-Agent': 'RandomDotOrg.py/%s + %s' % (__version__, __url__)}
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req).read().splitlines()


class RandomDotOrg(random.Random):
    """Alternate random number generator using random.org http
    service as source.

    RANDOM.ORG is a true random number service that generates randomness
    via atmospheric noise.
    """

    #--- New methods
    def get_quota(self):
        """
        Returns used bit quota
        """
        return int(_fetch_randomorg('quota')[0])

    def get_seed(self):
        """Returns a really random seed suitable to use with random module.
        It will be a 20 digit long integer.

        >>> import random, randomdotorg
        >>> random.seed(randomdotorg.RandomDotOrg().get_seed())
        """
        intlist =  _fetch_randomorg('integers', num=4, max=99999)
        return int(''.join(number.rjust(5, '0') for number in intlist))

    def write_random_bytes(self, fileobj, num_bytes=256):
        """Writes the specified number of bytes to the file object passed.
        Use it to feed /dev/random:
        >>> r = randomdotorg.RandomDotOrg()
        >>> r.write_random_bytes(open('/dev/random', 'a'))
        """
        for num in self.randrange(256, ammount=num_bytes):
            fileobj.write(chr(num))

    #--- Required overwritten methods
    def random(self, ammount=None):
        """Get a random number in the range [0.0, 1.0).

        if the `ammount` parameter is not None (the default), returns a list of
        `ammount` results, but will efficiently fetch multiple numbers at
        a single request.
        """
        if ammount is None:
            nints = 5
        else:
            nints = ammount * 5
        pool = _fetch_randomorg('integers', num=nints, max=999)
        grouped = (pool[i:i+5] for i in range(0, nints, 5))
        result = [float('0.%s' % ''.join(number.rjust(3, '0') for number in intlist))
                  for intlist in grouped]
        if ammount is None:
            return result[0]
        else:
            return result

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates a long int with k random bits."""
        k = int(k)
        if k <= 0:
            raise ValueError('number of bits must be greater than zero')
        bits = _fetch_randomorg('integers', num=k, max=1, base=2)
        return int(''.join(bits), 2)

    #--- Stub & Not implemented methods
    def _stub(self, *args, **kwds):
        "Stub method. Not used for a random.org random number generator."
        return None
    seed = jumpahead = _stub

    def _notimplemented(self, *args, **kwds):
        "Method should not be called for a random.org number generator."
        raise NotImplementedError('Random.org entropy source state saving is not implemented.')
    getstate = setstate = _notimplemented

    #--- Methods reimplemented to save bit quota (each .random() spends 50 bits)
    def shuffle(self, l):
        """l -> shuffle list l in place; return None.
        """
        order = _fetch_randomorg('sequences', max=len(l) - 1)
        for index, content in enumerate(l[:]):
            l[int(order[index])] = content

    def choice(self, seq, ammount=None):
        """Choose a random element from a non-empty sequence.

        if the `ammount` parameter is not None (the default), returns a list of
        `ammount` results, but will efficiently fetch multiple numbers at
        a single request.
        """
        if ammount is None:
            nints = 1
        else:
            nints = ammount
        n = len(seq)
        if n == 0:
            results = [None]
        elif n == 1:
            results = [seq[0]]
        else:
            results = [seq[pos] for pos in
                       self.randrange(0, n, ammount=nints)]
        if ammount is None:
            return results[0]
        else:
            return results

    def sample(self, population, k):
        """Chooses k unique random elements from a population sequence."""
        n = len(population)
        if not 0 <= k <= n:
            raise ValueError("sample larger than population")
        order = _fetch_randomorg('sequences', max=n - 1)
        result = [population[int(order[n])] for n in range(k)]
        return result

    def randrange(self, start, stop=None, step=1, ammount=None):
        """Choose a random item from range([start,] stop[, step])

        if the `ammount` parameter is not None (the default), returns a list of
        `ammount` results, but will efficiently fetch multiple numbers at
        a single request.
        """
        if stop is None:
            start, stop = 0, start
        xr = range(start, stop, step)
        n = len(xr)
        if n == 0:
            raise ValueError("range is empty")
        if ammount is None:
            nints = 1
        else:
            nints = ammount
        positions = _fetch_randomorg('integers', num=nints, max=n - 1)
        result = [xr[int(pos)] for pos in positions]
        if ammount is None:
            return result[0]
        else:
            return result
