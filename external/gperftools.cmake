if(ENABLE_GPERFTOOLS OR ENABLE_TCMALLOC_MINIMAL)
  
  if(ENABLE_GPERFTOOLS)
    find_package(Gperftools COMPONENTS tcmalloc OPTIONAL_COMPONENTS profiler)
  else()
    find_package(Gperftools REQUIRED COMPONENTS tcmalloc_minimal)
  endif()

  # Set the config.h variables
  if(GPERFTOOLS_FOUND)
    set(MADNESS_HAS_GOOGLE_PERF 1)
    if (Gperftools_tcmalloc_FOUND OR Gperftools_tcmalloc_minimal_FOUND OR Gperftools_tcmalloc_and_profiler_FOUND)
      set(MADNESS_HAS_GOOGLE_PERF_TCMALLOC 1)
    endif(Gperftools_tcmalloc_FOUND OR Gperftools_tcmalloc_minimal_FOUND OR Gperftools_tcmalloc_and_profiler_FOUND)
    if (Gperftools_tcmalloc_minimal_FOUND)
      set(MADNESS_HAS_GOOGLE_PERF_TCMALLOC_MINIMAL 1)
    endif(Gperftools_tcmalloc_minimal_FOUND)
    if (Gperftools_profiler_FOUND OR Gperftools_tcmalloc_and_profiler_FOUND)
      set(MADNESS_HAS_GOOGLE_PERF_PROFILER 1)
    endif(Gperftools_profiler_FOUND OR Gperftools_tcmalloc_and_profiler_FOUND)
  endif()
  if(LIBUNWIND_FOUND)
    set(MADNESS_HAS_LIBUNWIND 1)
  endif()
      
endif()