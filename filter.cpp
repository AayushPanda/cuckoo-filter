#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdlib>

class Filter{
private:
    long *arr;
    size_t arr_size;
    static const size_t max_kicks = 500;
    static const int resize_factor = 2;

    long h1(long long x){
        const long A = 2654435769;
        return (A * x) % arr_size;
    }

    long h2(long long i, long long fingerprint){
        // We call this with x = fingerprint and x
        // So to calculate other index, we do i2 = i ^ fingerprint, where i = h1(x)
        // or to calculate i2, we do i1 = i2 ^ fingerprint, where i = h2(i1, fingerprint)
        return i ^ fingerprint;
    }

    void resize(){
        size_t old_size = arr_size;
        long *old_arr = arr;
        arr_size *= resize_factor;
        arr = new long[arr_size];
        for(size_t i = 0; i < arr_size; i++){
            arr[i] = -1;
        }
        for(size_t i = 0; i < old_size; i++){
            if(old_arr[i] != -1){
                insert(old_arr[i]);
            }
        }
        delete[] old_arr;
    }

public:
    Filter(size_t arr_size = 1<<5): arr_size(arr_size), arr(new long[arr_size]){
        for(size_t i = 0; i < arr_size; i++){
            arr[i] = -1;
        }
    }

    ~Filter(){
        delete[] arr;
    }
    
    void insert(long long x){
        for(size_t i = 0; i < max_kicks; i++){
            long fingerprint = x % (arr_size - 1) + 1; // ensure fingerprint not zero. if it is, then both h1 and h2 will return same index
            long i1 = h1(x);
            if(arr[i1] == -1){
                arr[i1] = fingerprint;
                return;
            }
            long i2 = h2(i1, fingerprint);
            if(arr[i2] == -1){
                arr[i2] = fingerprint;
                return;
            }
            // kick out
            if(rand() % 2 == 0){
                std::swap(fingerprint, arr[i1]);
                x = fingerprint;
            } else {
                std::swap(fingerprint, arr[i2]);
                x = fingerprint;
            }
        }

        resize();
        insert(x);

    }

    bool contains(long long x){
        long fingerprint = x % (arr_size - 1) + 1;
        long i1 = h1(x);
        if(arr[i1] == fingerprint){
            return true;
        }
        long i2 = h2(i1, fingerprint);
        if(arr[i2] == fingerprint){
            return true;
        }
        return false;   
    }
};


long find_inverse(long a, long m) {
    long m0 = m, t, q;
    long x0 = 0, x1 = 1;

    if (m == 1)
        return 0;

    while (a > 1) {
        q = a / m;
        t = m;

        m = a % m, a = t;
        t = x0;

        x0 = x1 - q * x0;
        x1 = t;
    }

    if (x1 < 0)
        x1 += m0;

    return x1;
}

long power_in_modulus(long x, long p, long m){
    // Take modulus regularly to prevent overflow
    long result = 1;
    x = x % m;
    while(p > 0){
        if(p & 1){
            result = (result * x) % m;
        }
        x = (x * x) % m;
        p >>= 1;
    }
    return result;
}

std::vector<std::pair<std::string_view, long>>  get_ngram_fingerprints(const std::string &text, size_t n_min, size_t n_max){
    // we calculate fingerprints of all n-grams where n_min <= n <= n_max
    // fingerprint of an ngram is calculated as the sum of characters in a polynomial rolling hash within a prime modulus
    // we first calculate a prefix sum array of character values, and then for each n-gram, we can get its fingerprint in O(1) time
    // to normalise window position, we divide the fingerprint by the multiplicative inverse of the base^(n-1) mod modulus
    // e.g. for n=3, the n-gram "abc" has fingerprint (a*b^2 + b*b^1 + c*b^0) (mod p)
    // and within in ab has the fingerprint (a*b^1 + b*b^0) (mod p) = (a*b^2 + b*b^1 + c*b^0 - c*b^0) * b^-1 (mod p)
    // this is to shift the window to the left by as many positions required so the strings are comparable and not affected by their position in the text

    constexpr const long base = 257;
    constexpr const long modulus = 1000000007;
    const long base_inv = find_inverse(base, modulus);

    size_t text_length = text.length();
    std::vector<long> prefix_sums(text_length + 1, 0);
    for(size_t i = 0; i < text_length; i++){
        prefix_sums[i + 1] = (prefix_sums[i] * base + static_cast<long>(text[i])) % modulus;
    }
    std::vector<std::pair<std::string_view, long>> ngram_fingerprints;
    for(size_t start = 0; start < text_length; ++start){
        for(size_t end = start + n_min; end <= std::min(text_length, start + n_max); ++end){
            std::string_view sv = std::string_view(text).substr(start, end - start);
            long fingerprint = ((prefix_sums[end] - prefix_sums[start] + modulus) % modulus * power_in_modulus(base_inv, start, modulus)) % modulus;
            ngram_fingerprints.push_back(std::pair(sv, fingerprint));
        }
    }

    return ngram_fingerprints;
}

std::vector<std::string_view> get_ngrams_in_set(const std::string &text, Filter f, size_t min_ngram_size, size_t max_ngram_size){
    auto ngrams = get_ngram_fingerprints(text, min_ngram_size, max_ngram_size);
    std::vector<std::string_view> results;

    for(auto ngram: ngrams){
        if (f.contains(ngram.second)){
            results.push_back(ngram.first);
        }
    }

    return results;
}


namespace py = pybind11;

PYBIND11_MODULE(filter_module, m) {
    py::class_<Filter>(m, "Filter")
        .def(py::init<size_t>(), py::arg("arr_size") = 1<<5)
        .def("insert", &Filter::insert)
        .def("contains", &Filter::contains);
    
    m.def("get_ngrams_in_set", &get_ngrams_in_set);
}
