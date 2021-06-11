cdef extern from "parallel_hashmap/phmap.h" namespace "phmap":
    cdef cppclass flat_hash_map[K, V]:
        K& key_type
        V& mapped_type
        flat_hash_map()
        V& operator[](K&) nogil
