#include <x86intrin.h>
#include <vector>
#include <array>
#include <iostream>
#include <random>
#include <chrono>

namespace {
float trace(float m[9]) {
  return m[0] + m[4] + m[8];
}

__m128 trace(
    __m128 m00, __m128 m13, __m128 m26,
                __m128 m44, __m128 m57,
                            __m128 m88) {
  return _mm_add_ps(m00, _mm_add_ps(m44, m88));
}

float trace_square(float m[9]) {
  float tr = 0.0f;
  for (int i=0; i<3; ++i)
    for (int k=0; k<3; ++k)
      tr += m[i*3+k]*m[k*3+i];
  return tr;
}

__m128 trace_square(
    __m128 m00, __m128 m13, __m128 m26,
                __m128 m44, __m128 m57,
                            __m128 m88) {
  return _mm_add_ps(
    _mm_add_ps(
      _mm_add_ps(
        _mm_add_ps(
          _mm_mul_ps(m00, m00),
          _mm_mul_ps(m13, m13)),
        _mm_add_ps(
          _mm_mul_ps(m26, m26),
          _mm_mul_ps(m13, m13))),
      _mm_add_ps(
        _mm_add_ps(
          _mm_mul_ps(m44, m44),
          _mm_mul_ps(m57, m57)),
        _mm_add_ps(
          _mm_mul_ps(m26, m26),
          _mm_mul_ps(m57, m57)))),
    _mm_mul_ps(m88, m88));
}

float det(float m[9]) {
  return m[0]*(m[4]*m[8]-m[5]*m[7]) - m[3]*(m[1]*m[8]-m[2]*m[7]) + m[6]*(m[1]*m[5]-m[2]*m[4]);
}

__m128 det(
    __m128 m00, __m128 m13, __m128 m26,
                __m128 m44, __m128 m57,
                            __m128 m88) {
  __m128 vdet;
  vdet = _mm_add_ps(
      _mm_mul_ps(m00, _mm_sub_ps(_mm_mul_ps(m44, m88), _mm_mul_ps(m57, m57))),
      _mm_mul_ps(m26, _mm_sub_ps(_mm_mul_ps(m13, m57), _mm_mul_ps(m26, m44))));
  vdet = _mm_sub_ps(vdet, _mm_mul_ps(m13, _mm_sub_ps(_mm_mul_ps(m13, m88), _mm_mul_ps(m26, m57))));
  return vdet;
}

template <size_t newton_iter, int version>
void cubic_eigen(float m[9], float e[3]) {
  if (version == 0) {
    const float a = -trace(m);
    const float b = -0.5f*(trace_square(m) - a*a);
    const float c = -det(m);
    const float sqrt = std::sqrt(a*a - 3.0f*b);
    const float r = -a/3.0f - sqrt;
    const float s = -a/3.0f;
    const float t = -a/3.0f + sqrt;
    auto newton = [=](float x) {
      for (size_t i=0; i<newton_iter; ++i) {
        x = x - (x*x*x + a*x*x + b*x + c)/(3*x*x + 2*a*x + b);
      }
      return x;
    };
    const float x_r = newton(r);
    const float x_s = newton(s);
    const float x_t = newton(t);
    e[0] = x_r;
    e[1] = x_s;
    e[2] = x_t;
  }
  if (version == 1) {
    const float a = -trace(m);
    const float b = -0.5f*(trace_square(m) - a*a);
    const float c = -det(m);
    const float sqrt = std::sqrt(a*a - 3.0f*b);
    const float h = -a/3.0f;
    const float t = -a/3.0f + (h*h*h + a*h*h + b*h + c > 0 ? sqrt : - sqrt);
    auto newton = [=](float x) {
      for (size_t i=0; i<newton_iter; ++i) {
        x = x - (x*x*x + a*x*x + b*x + c)/(3*x*x + 2*a*x + b);
      }
      return x;
    };
    float x_t = newton(t);
    const float alpha = a + x_t;
    const float beta  = b + alpha*x_t;
    const float sqrt2 = std::sqrt(alpha*alpha - 4.0f*beta);
    float x_r = (-alpha - sqrt2)*0.5f;
    float x_s = (-alpha + sqrt2)*0.5f;
    if (x_t < x_r) std::swap(x_t, x_r);
    if (x_t < x_s) std::swap(x_t, x_s);
    e[0] = x_r;
    e[1] = x_s;
    e[2] = x_t;
  }
}

template <size_t newton_iter, int version>
void cubic_eigen(
    __m128 m00, __m128 m13, __m128 m26,
                __m128 m44, __m128 m57,
                            __m128 m88,
    __m128& e0, __m128& e1, __m128& e2) {
  if (version == 0) {
#define M m00, m13, m26, m44, m57, m88
    const __m128 vaneg = trace(M);
    const __m128 va2 = _mm_mul_ps(vaneg, vaneg);
    const __m128 vb = _mm_mul_ps(_mm_set1_ps(-0.5f), _mm_sub_ps(trace_square(M), va2));
    const __m128 vcneg = det(M);
    const __m128 vsqrt = _mm_sqrt_ps(_mm_sub_ps(va2, _mm_mul_ps(_mm_set1_ps(3.0f), vb)));
    const __m128 vs = _mm_mul_ps(vaneg, _mm_set1_ps(1.0f/3.0f));
    const __m128 vr = _mm_sub_ps(vs, vsqrt);
    const __m128 vt = _mm_add_ps(vs, vsqrt);
#undef M
    auto newton = [=](__m128 vx) {
      for (size_t i=0; i<newton_iter; ++i) {
        __m128 vx2 = _mm_mul_ps(vx, vx);
        __m128 vaneg_mult_vx = _mm_mul_ps(vaneg, vx);
        vx = _mm_sub_ps(vx, _mm_mul_ps(_mm_sub_ps(
              _mm_mul_ps(_mm_add_ps(
                  _mm_sub_ps(vx2, vaneg_mult_vx),
                  vb), vx), vcneg),
              _mm_rcp_ps(
                _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(3.0f), vx2),
                    _mm_add_ps(vaneg_mult_vx, vaneg_mult_vx)),
                  vb)
                )));
      }
      return vx;
    };
    e0 = newton(vr);
    e1 = newton(vs);
    e2 = newton(vt);
  }
  if (version == 1) {
#define M m00, m13, m26, m44, m57, m88
    const __m128 vaneg = trace(M);
    const __m128 va2 = _mm_mul_ps(vaneg, vaneg);
    const __m128 vb = _mm_mul_ps(_mm_set1_ps(-0.5f), _mm_sub_ps(trace_square(M), va2));
    const __m128 vcneg = det(M);
    const __m128 vsqrt = _mm_sqrt_ps(_mm_sub_ps(va2, _mm_mul_ps(_mm_set1_ps(3.0f), vb)));
    const __m128 vh = _mm_mul_ps(vaneg, _mm_set1_ps(1.0f/3.0f));
    const __m128 vfh = _mm_sub_ps(_mm_mul_ps(vh, _mm_add_ps(_mm_mul_ps(vh, _mm_sub_ps(vh, vaneg)), vb)), vcneg);
    const __m128 vmask = _mm_cmpgt_ps(vfh, _mm_setzero_ps());
    const __m128 vtstart = _mm_add_ps(vh, _mm_or_ps(_mm_and_ps(vmask, vsqrt), _mm_andnot_ps(vmask, _mm_sub_ps(_mm_setzero_ps(), vsqrt))));
#undef M
    auto newton = [=](__m128 vx) {
      for (size_t i=0; i<newton_iter; ++i) {
        __m128 vx2 = _mm_mul_ps(vx, vx);
        __m128 vaneg_mult_vx = _mm_mul_ps(vaneg, vx);
        vx = _mm_sub_ps(vx, _mm_mul_ps(_mm_sub_ps(
              _mm_mul_ps(_mm_add_ps(
                  _mm_sub_ps(vx2, vaneg_mult_vx),
                  vb), vx), vcneg),
              _mm_rcp_ps(
                _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(3.0f), vx2),
                    _mm_add_ps(vaneg_mult_vx, vaneg_mult_vx)),
                  vb)
                )));
      }
      return vx;
    };
    __m128 vt = newton(vtstart);
    __m128 valphaneg = _mm_sub_ps(vaneg, vt);
    __m128 vbeta = _mm_sub_ps(vb, _mm_mul_ps(valphaneg, vt));
    __m128 valphaneghalf = _mm_mul_ps(valphaneg, _mm_set1_ps(0.5f));
    __m128 vsqrt2 = _mm_sqrt_ps(_mm_sub_ps(_mm_mul_ps(valphaneghalf, valphaneghalf), vbeta));
    __m128 vr = _mm_sub_ps(valphaneghalf, vsqrt2);
    __m128 vs = _mm_add_ps(valphaneghalf, vsqrt2);
    __m128 vtrmax = _mm_max_ps(vt, vr);
    e0 = _mm_min_ps(vt, vr);
    e1 = _mm_min_ps(vtrmax, vs);
    e2 = _mm_max_ps(vtrmax, vs);
  }
}
} // anonymous namespace

// calculate the eigenvalues of the Hessian of the volume.
// e0 <= e1 <= e2
template <size_t newton_iter, int version>
void eigen_hessian_3d(float* dst_e0, float* dst_e1, float* dst_e2, const float* src, size_t w, size_t h, size_t d, bool ref) {
  auto v = [=](size_t x, size_t y, size_t z) -> const float& { return src[(z*h + y)*w + x]; };
  auto ev = [=](float* vol, size_t x, size_t y, size_t z) -> float& { return vol[(z*h + y)*w + x]; };
  for (size_t z=1; z<d-1; ++z) {
    for (size_t y=1; y<h-1; ++y) {
      auto ref_pixel = [=](size_t x) {
        const float f = 2.0f*v(x, y, z);
        const float f_xx = v(x + 1, y, z) + v(x - 1, y, z) - f;
        const float f_yy = v(x, y + 1, z) + v(x, y - 1, z) - f;
        const float f_zz = v(x, y, z + 1) + v(x, y, z - 1) - f;
        const float f_xy = v(x + 1, y + 1, z) + v(x - 1, y - 1, z) - v(x - 1, y + 1, z) - v(x + 1, y - 1, z);
        const float f_yz = v(x, y + 1, z + 1) + v(x, y - 1, z - 1) - v(x, y - 1, z + 1) - v(x, y + 1, z - 1);
        const float f_zx = v(x + 1, y, z + 1) + v(x - 1, y, z - 1) - v(x + 1, y, z - 1) - v(x - 1, y, z + 1);
        float m[9] = {
          f_xx, f_xy, f_zx,
          f_xy, f_yy, f_yz,
          f_zx, f_yz, f_zz};
        float eigen[3];
        cubic_eigen<newton_iter, version>(m, eigen);
        ev(dst_e0, x, y, z) = eigen[0];
        ev(dst_e1, x, y, z) = eigen[1];
        ev(dst_e2, x, y, z) = eigen[2];
      };
      auto simd_pixel = [=](size_t x) {
        __m128 vf2 = _mm_loadu_ps((const float*)&src[(z*h + y)*w + x]);
        vf2 = _mm_add_ps(vf2, vf2);
        __m128 vfxm1 = _mm_loadu_ps((const float*)&src[(z*h + y)*w + x - 1]);
        __m128 vfxp1 = _mm_loadu_ps((const float*)&src[(z*h + y)*w + x + 1]);
        __m128 vfym1 = _mm_loadu_ps((const float*)&src[(z*h + y - 1)*w + x]);
        __m128 vfyp1 = _mm_loadu_ps((const float*)&src[(z*h + y + 1)*w + x]);
        __m128 vfzm1 = _mm_loadu_ps((const float*)&src[((z - 1)*h + y)*w + x]);
        __m128 vfzp1 = _mm_loadu_ps((const float*)&src[((z + 1)*h + y)*w + x]);
        __m128 vfxx = _mm_sub_ps(_mm_add_ps(vfxm1, vfxp1), vf2);
        __m128 vfyy = _mm_sub_ps(_mm_add_ps(vfym1, vfyp1), vf2);
        __m128 vfzz = _mm_sub_ps(_mm_add_ps(vfzm1, vfzp1), vf2);
        __m128 vfxm1ym1 = _mm_loadu_ps((const float*)&src[(z*h + y - 1)*w + x - 1]);
        __m128 vfxp1ym1 = _mm_loadu_ps((const float*)&src[(z*h + y - 1)*w + x + 1]);
        __m128 vfxm1yp1 = _mm_loadu_ps((const float*)&src[(z*h + y + 1)*w + x - 1]);
        __m128 vfxp1yp1 = _mm_loadu_ps((const float*)&src[(z*h + y + 1)*w + x + 1]);
        __m128 vfym1zm1 = _mm_loadu_ps((const float*)&src[((z - 1)*h + y - 1)*w + x]);
        __m128 vfyp1zm1 = _mm_loadu_ps((const float*)&src[((z - 1)*h + y + 1)*w + x]);
        __m128 vfym1zp1 = _mm_loadu_ps((const float*)&src[((z + 1)*h + y - 1)*w + x]);
        __m128 vfyp1zp1 = _mm_loadu_ps((const float*)&src[((z + 1)*h + y + 1)*w + x]);
        __m128 vfzm1xm1 = _mm_loadu_ps((const float*)&src[((z - 1)*h + y)*w + x - 1]);
        __m128 vfzp1xm1 = _mm_loadu_ps((const float*)&src[((z + 1)*h + y)*w + x - 1]);
        __m128 vfzm1xp1 = _mm_loadu_ps((const float*)&src[((z - 1)*h + y)*w + x + 1]);
        __m128 vfzp1xp1 = _mm_loadu_ps((const float*)&src[((z + 1)*h + y)*w + x + 1]);
        __m128 vfxy = _mm_sub_ps(_mm_add_ps(vfxp1yp1, vfxm1ym1), _mm_add_ps(vfxm1yp1, vfxp1ym1));
        __m128 vfyz = _mm_sub_ps(_mm_add_ps(vfyp1zp1, vfym1zm1), _mm_add_ps(vfym1zp1, vfyp1zm1));
        __m128 vfzx = _mm_sub_ps(_mm_add_ps(vfzp1xp1, vfzm1xm1), _mm_add_ps(vfzm1xp1, vfzp1xm1));
        __m128 ve0, ve1, ve2;
        cubic_eigen<newton_iter, version>(vfxx, vfxy, vfzx, vfyy, vfyz, vfzz, ve0, ve1, ve2);
        _mm_storeu_ps((float*)&dst_e0[(z*h + y)*w + x], ve0);
        _mm_storeu_ps((float*)&dst_e1[(z*h + y)*w + x], ve1);
        _mm_storeu_ps((float*)&dst_e2[(z*h + y)*w + x], ve2);
      };
      size_t x = 1;
      for (; x<std::min<size_t>(w-1, 4); ++x)
        ref_pixel(x);
      if (!ref) {
        for (; x<w-1-4; x+=4)
          simd_pixel(x);
      }
      for (; x<w-1; ++x)
        ref_pixel(x);
    }
  }
}

struct eigen_result {
  std::vector<float> e0;
  std::vector<float> e1;
  std::vector<float> e2;
  size_t newton_iter;
  eigen_result(size_t N)
   : e0(N*N*N, 0.0f)
   , e1(N*N*N, 0.0f)
   , e2(N*N*N, 0.0f)
  , newton_iter(0) {}
};
void show_diff(const eigen_result& ra, const eigen_result& rb, bool verbose) {
  float e0_diff_sum = 0.0f;
  float e1_diff_sum = 0.0f;
  float e2_diff_sum = 0.0f;
  for (size_t i=0; i<ra.e0.size(); ++i) {
    if (verbose) {
      std::cout << ""
                << ra.e0[i] << ","
                << ra.e1[i] << ","
                << ra.e2[i] << " "
                << rb.e0[i] << ","
                << rb.e1[i] << ","
                << rb.e2[i] << " "
                << std::endl;
    }
    e0_diff_sum += std::abs(ra.e0[i] - rb.e0[i]);
    e1_diff_sum += std::abs(ra.e1[i] - rb.e1[i]);
    e2_diff_sum += std::abs(ra.e2[i] - rb.e2[i]);
  }
  std::cout << "newton_iter = (" << ra.newton_iter << ", " << rb.newton_iter << ") mean abs error: "
    << e0_diff_sum/ra.e0.size() << ", "
    << e1_diff_sum/ra.e1.size() << ", "
    << e2_diff_sum/ra.e2.size() << std::endl;
}

template <typename Func>
void timer(std::string label, size_t n_iter, Func&& f) {
  auto t_start = std::chrono::system_clock::now();
  for (size_t i=0; i<n_iter; ++i) {
    f();
  }
  auto t_stop = std::chrono::system_clock::now();
  std::cout << label << ": "
    << float(std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start).count()) / float(n_iter) << " ms/iter"
    << std::endl;
}

int main() {
  const size_t N = 64;
  const size_t n_iter = 20;
  const size_t newton_iter = 4;
  const bool verbose = false;

  std::vector<float> volume(N*N*N, 0.0f);
  std::mt19937 mt(0);
  std::uniform_real_distribution<> uniform(-1.0f, 1.0f);
  for (size_t i=0; i<volume.size(); ++i) {
    volume[i] = uniform(mt);
  }

  eigen_result ref0(N);
  eigen_result ref1(N);
  timer("reference0", n_iter, [&]{
    ref0.newton_iter = newton_iter;
    eigen_hessian_3d<newton_iter, 0>(ref0.e0.data(), ref0.e1.data(), ref0.e2.data(), volume.data(), N, N, N, true);
  });
  timer("reference1", n_iter, [&]{
    ref1.newton_iter = newton_iter;
    eigen_hessian_3d<newton_iter, 1>(ref1.e0.data(), ref1.e1.data(), ref1.e2.data(), volume.data(), N, N, N, true);
  });

  eigen_result simd0(N);
  eigen_result simd1(N);
  timer("SSE0-1", n_iter, [&]{
    simd0.newton_iter = 1;
    eigen_hessian_3d<1, 0>(simd0.e0.data(), simd0.e1.data(), simd0.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE0-2", n_iter, [&]{
    simd0.newton_iter = 2;
    eigen_hessian_3d<2, 0>(simd0.e0.data(), simd0.e1.data(), simd0.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE0-3", n_iter, [&]{
    simd0.newton_iter = 3;
    eigen_hessian_3d<3, 0>(simd0.e0.data(), simd0.e1.data(), simd0.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE0", n_iter, [&]{
    simd0.newton_iter = newton_iter;
    eigen_hessian_3d<newton_iter, 0>(simd0.e0.data(), simd0.e1.data(), simd0.e2.data(), volume.data(), N, N, N, false);
  });

  timer("SSE1-1", n_iter, [&]{
    simd1.newton_iter = 1;
    eigen_hessian_3d<1, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1-2", n_iter, [&]{
    simd1.newton_iter = 2;
    eigen_hessian_3d<2, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1-3", n_iter, [&]{
    simd1.newton_iter = 3;
    eigen_hessian_3d<3, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1-4", n_iter, [&]{
    simd1.newton_iter = 4;
    eigen_hessian_3d<4, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1-5", n_iter, [&]{
    simd1.newton_iter = 5;
    eigen_hessian_3d<5, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1-6", n_iter, [&]{
    simd1.newton_iter = 6;
    eigen_hessian_3d<6, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });
  timer("SSE1", n_iter, [&]{
    simd1.newton_iter = newton_iter;
    eigen_hessian_3d<newton_iter, 1>(simd1.e0.data(), simd1.e1.data(), simd1.e2.data(), volume.data(), N, N, N, false);
  });

  std::cout << "ref0 ~ ref1: " << N << "x" << N << "x" << N << " volume. : ";
  show_diff(ref0, ref1, verbose);
  std::cout << "ref0 ~ simd0: " << N << "x" << N << "x" << N << " volume. : ";
  show_diff(ref0, simd0, verbose);
  std::cout << "ref1 ~ simd1: " << N << "x" << N << "x" << N << " volume. : ";
  show_diff(ref1, simd1, verbose);
  std::cout << "simd0 ~ simd1: " << N << "x" << N << "x" << N << " volume. : ";
  show_diff(simd0, simd1, verbose);
  std::cout << "ref0 ~ simd1: " << N << "x" << N << "x" << N << " volume. : ";
  show_diff(ref0, simd1, verbose);

  return 0;
}

