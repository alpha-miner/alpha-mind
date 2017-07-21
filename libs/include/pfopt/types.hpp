#ifndef pfopt_types_hpp
#define pfopt_types_hpp

#include <Eigen/Eigen>
#include <ap.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using alglib::real_1d_array;
using alglib::real_2d_array;


#ifdef  WIN32 
#ifdef __cplusplus 
#define DLL_EXPORT_C_DECL extern "C" __declspec(dllexport) 
#define DLL_IMPORT_C_DECL extern "C" __declspec(dllimport) 
#define DLL_EXPORT_DECL extern __declspec(dllexport) 
#define DLL_IMPORT_DECL extern __declspec(dllimport) 
#define DLL_EXPORT_CLASS_DECL __declspec(dllexport) 
#define DLL_IMPORT_CLASS_DECL __declspec(dllimport) 
#else 
#define DLL_EXPORT_DECL __declspec(dllexport) 
#define DLL_IMPORT_DECL __declspec(dllimport) 
#endif 
#else 
#ifdef __cplusplus 
#define DLL_EXPORT_C_DECL extern "C" 
#define DLL_IMPORT_C_DECL extern "C" 
#define DLL_EXPORT_DECL extern 
#define DLL_IMPORT_DECL extern 
#define DLL_EXPORT_CLASS_DECL 
#define DLL_IMPORT_CLASS_DECL 
#else 
#define DLL_EXPORT_DECL extern 
#define DLL_IMPORT_DECL extern 
#endif 
#endif

#ifdef PFOPF_EXPORTS 
#define PFOPT_CLASS DLL_EXPORT_CLASS_DECL 
#define PFOPT_API DLL_EXPORT_DECL 
#else 
#define PFOPT_CLASS DLL_IMPORT_CLASS_DECL 
#define PFOPT_API DLL_IMPORT_DECL 
#endif 


#endif
