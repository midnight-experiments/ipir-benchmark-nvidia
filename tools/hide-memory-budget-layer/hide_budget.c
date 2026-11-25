// Vulkan layer VK_LAYER_IPIRBENCH_hide_memory_budget: removes VK_EXT_memory_budget from the device extension list, so
// wgpu does not enable it and skips its memory-budget checks. Everything else passes straight through.
//
// Why: without resizable BAR, NVIDIA's driver reports the BAR heap (DEVICE_LOCAL | HOST_VISIBLE, 246 MiB on an
// RTX 2060) with heapUsage = all processes' usage and heapBudget = heap size - that usage. wgpu 28 (Deno 2.7.5:
// for_resource_creation 97 %, for_device_loss 99 %) refuses every buffer once usage + size >= 97 % of budget on any
// device-local heap, and loses the device once usage >= 99 % of budget. With a desktop holding ~88 MiB of BAR and the
// device's own 64 MiB staging block, every allocation fails ("not enough memory left") and the device is lost.
//
// The loader-layer structs are declared here (vk_layer.h layout), so the layer builds without the Vulkan headers.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*PFN_vkVoidFunction)(void);
typedef void *VkInstance;
typedef void *VkPhysicalDevice;
typedef void *VkDevice;
typedef int VkResult;
typedef PFN_vkVoidFunction (*PFN_GIPA)(VkInstance, const char *);
typedef PFN_vkVoidFunction (*PFN_GDPA)(VkDevice, const char *);

typedef struct { char extensionName[256]; uint32_t specVersion; } VkExtensionProperties;
typedef struct { int sType; const void *pNext; } BaseIn;

typedef struct InstLink { struct InstLink *pNext; PFN_GIPA nextGIPA; void *nextGPDPA; } InstLink;
typedef struct { int sType; const void *pNext; int function; union { InstLink *pLayerInfo; void *other; } u; } LayerInstanceCreateInfo;
typedef struct DevLink { struct DevLink *pNext; PFN_GIPA nextGIPA; PFN_GDPA nextGDPA; } DevLink;
typedef struct { int sType; const void *pNext; int function; union { DevLink *pLayerInfo; void *other; } u; } LayerDeviceCreateInfo;

#define STYPE_LOADER_INSTANCE_CREATE_INFO 47
#define STYPE_LOADER_DEVICE_CREATE_INFO 48
#define LAYER_LINK_INFO 0
#define VK_INCOMPLETE 5
#define LAYER_NAME "VK_LAYER_IPIRBENCH_hide_memory_budget"
#define HIDDEN "VK_EXT_memory_budget"
#define EXPORT __attribute__((visibility("default")))

static PFN_GIPA next_gipa;
static VkInstance the_instance;
#define MAX_DEVICES 16
static struct { VkDevice device; PFN_GDPA gdpa; } devices[MAX_DEVICES];

EXPORT PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char *name);
EXPORT PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, const char *name);

static VkResult layer_CreateInstance(const void *info, const void *alloc, VkInstance *out) {
  LayerInstanceCreateInfo *chain = (LayerInstanceCreateInfo *)((const BaseIn *)info)->pNext;
  while (chain && !(chain->sType == STYPE_LOADER_INSTANCE_CREATE_INFO && chain->function == LAYER_LINK_INFO))
    chain = (LayerInstanceCreateInfo *)chain->pNext;
  if (!chain) return -3; // VK_ERROR_INITIALIZATION_FAILED
  PFN_GIPA gipa = chain->u.pLayerInfo->nextGIPA;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  VkResult (*create)(const void *, const void *, VkInstance *) = (void *)gipa(NULL, "vkCreateInstance");
  VkResult r = create(info, alloc, out);
  if (r == 0) { next_gipa = gipa; the_instance = *out; }
  return r;
}

static VkResult layer_CreateDevice(VkPhysicalDevice pd, const void *info, const void *alloc, VkDevice *out) {
  LayerDeviceCreateInfo *chain = (LayerDeviceCreateInfo *)((const BaseIn *)info)->pNext;
  while (chain && !(chain->sType == STYPE_LOADER_DEVICE_CREATE_INFO && chain->function == LAYER_LINK_INFO))
    chain = (LayerDeviceCreateInfo *)chain->pNext;
  if (!chain) return -3;
  PFN_GIPA gipa = chain->u.pLayerInfo->nextGIPA;
  PFN_GDPA gdpa = chain->u.pLayerInfo->nextGDPA;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  VkResult (*create)(VkPhysicalDevice, const void *, const void *, VkDevice *) = (void *)gipa(the_instance, "vkCreateDevice");
  VkResult r = create(pd, info, alloc, out);
  if (r == 0) {
    for (int i = 0; i < MAX_DEVICES; i++)
      if (!devices[i].device || devices[i].device == *out) { devices[i].device = *out; devices[i].gdpa = gdpa; break; }
  }
  return r;
}

static VkResult layer_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char *layer, uint32_t *count, VkExtensionProperties *props) {
  if (layer && strcmp(layer, LAYER_NAME) == 0) { *count = 0; return 0; }
  VkResult (*next)(VkPhysicalDevice, const char *, uint32_t *, VkExtensionProperties *) =
      (void *)next_gipa(the_instance, "vkEnumerateDeviceExtensionProperties");
  if (layer) return next(pd, layer, count, props);
  uint32_t n = 0;
  VkResult r = next(pd, NULL, &n, NULL);
  if (r != 0) return r;
  VkExtensionProperties *all = calloc(n ? n : 1, sizeof *all);
  r = next(pd, NULL, &n, all);
  if (r != 0 && r != VK_INCOMPLETE) { free(all); return r; }
  uint32_t kept = 0;
  for (uint32_t i = 0; i < n; i++)
    if (strcmp(all[i].extensionName, HIDDEN) != 0) all[kept++] = all[i];
  if (!props) { *count = kept; free(all); return 0; }
  uint32_t w = *count < kept ? *count : kept;
  memcpy(props, all, w * sizeof *all);
  *count = w;
  free(all);
  return w < kept ? VK_INCOMPLETE : 0;
}

EXPORT PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, const char *name) {
  if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
  for (int i = 0; i < MAX_DEVICES; i++)
    if (devices[i].device == device) return devices[i].gdpa(device, name);
  return NULL;
}

EXPORT PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char *name) {
  if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
  if (strcmp(name, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)layer_CreateInstance;
  if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)layer_CreateDevice;
  if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
  if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return (PFN_vkVoidFunction)layer_EnumerateDeviceExtensionProperties;
  return next_gipa ? next_gipa(instance, name) : NULL;
}
