/**
 * mlp_driver.h — MLP Accelerator Driver for ARM Cortex-A9
 * =======================================================
 *
 * Higher-level driver abstraction over the AXI GPIO interface.
 * Provides convenience functions for the MLP accelerator control.
 *
 * Usage:
 *   #include "mlp_driver.h"
 *   MLPDev mlp;
 *   MLP_Init(&mlp);
 *   MLP_SendImage(&mlp, my_image);
 *   int pred = MLP_Run(&mlp);
 */

#ifndef MLP_DRIVER_H
#define MLP_DRIVER_H

#include "xgpio.h"
#include "xparameters.h"
#include "sleep.h"

//=============================================================================
// MLP Accelerator Device Structure
//=============================================================================

typedef struct {
    XGpio gpio_ctrl;     // AXI GPIO for control (PS→PL)
    XGpio gpio_status;   // AXI GPIO for status (PL→PS)
    u32   baseaddr_ctrl;
    u32   baseaddr_status;
    int   initialized;
} MLPDev;

//=============================================================================
// Bit Field Definitions (match block design)
//=============================================================================

// Control register (GPIO channel 1)
#define MLP_CTRL_START_POS     0
#define MLP_CTRL_RST_POS       1
#define MLP_CTRL_START_MASK    (1 << MLP_CTRL_START_POS)
#define MLP_CTRL_RST_MASK      (1 << MLP_CTRL_RST_POS)

// Status register (GPIO channel 1)
#define MLP_STATUS_DONE_POS     0
#define MLP_STATUS_CLASS_POS    4
#define MLP_STATUS_DONE_MASK    (1 << MLP_STATUS_DONE_POS)
#define MLP_STATUS_CLASS_MASK   (0x0F << MLP_STATUS_CLASS_POS)

// Control channel numbers
#define MLP_CTRL_CH_SIGNAL     1   // start, rst bits
#define MLP_CTRL_CH_PIXEL      2   // pixel data
#define MLP_STATUS_CH          1   // done, predicted

// Timing
#define MLP_PIXEL_DELAY_US     1   // delay per pixel transfer
#define MLP_POLL_INTERVAL_US   10  // poll interval for done
#define MLP_TIMEOUT_US         100000  // 100 ms timeout

//=============================================================================
// Driver Functions
//=============================================================================

/**
 * Initialize the MLP accelerator driver.
 * Must be called before any other driver function.
 */
static inline int MLP_Init(MLPDev *dev)
{
    int status;

    status = XGpio_Initialize(&dev->gpio_ctrl, XPAR_AXI_GPIO_CTRL_DEVICE_ID);
    if (status != XST_SUCCESS) return status;

    status = XGpio_Initialize(&dev->gpio_status, XPAR_AXI_GPIO_STATUS_DEVICE_ID);
    if (status != XST_SUCCESS) return status;

    // Set directions
    XGpio_SetDataDirection(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL, 0x0);
    XGpio_SetDataDirection(&dev->gpio_ctrl, MLP_CTRL_CH_PIXEL, 0x0);
    XGpio_SetDataDirection(&dev->gpio_status, MLP_STATUS_CH, 0xFF);

    // Default values
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL, 0);
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_PIXEL, 0);

    dev->baseaddr_ctrl   = XPAR_AXI_GPIO_CTRL_BASEADDR;
    dev->baseaddr_status = XPAR_AXI_GPIO_STATUS_BASEADDR;
    dev->initialized     = 1;

    return XST_SUCCESS;
}

/**
 * Send a single pixel value to the accelerator.
 */
static inline void MLP_SendPixel(MLPDev *dev, short pixel_q88)
{
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_PIXEL,
                        (u32)(pixel_q88 & 0xFFFF));
}

/**
 * Send an entire image (784 pixels) to the accelerator.
 * Blocks until all pixels are transferred.
 */
static inline void MLP_SendImage(MLPDev *dev, short *image_q88)
{
    for (int i = 0; i < 784; i++) {
        MLP_SendPixel(dev, image_q88[i]);
        usleep(MLP_PIXEL_DELAY_US);
    }
}

/**
 * Start the inference by pulsing the start signal.
 */
static inline void MLP_Start(MLPDev *dev)
{
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL,
                        MLP_CTRL_START_MASK);
    usleep(1);
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL, 0);
}

/**
 * Check if the inference is complete.
 * Returns 1 if done, 0 otherwise.
 */
static inline int MLP_IsDone(MLPDev *dev)
{
    u32 status = XGpio_DiscreteRead(&dev->gpio_status, MLP_STATUS_CH);
    return (status & MLP_STATUS_DONE_MASK) ? 1 : 0;
}

/**
 * Wait for inference to complete with timeout.
 * Returns: 0 on success, -1 on timeout.
 */
static inline int MLP_WaitDone(MLPDev *dev)
{
    int elapsed = 0;
    while (!MLP_IsDone(dev)) {
        usleep(MLP_POLL_INTERVAL_US);
        elapsed += MLP_POLL_INTERVAL_US;
        if (elapsed > MLP_TIMEOUT_US) {
            return -1;  // timeout
        }
    }
    return elapsed;  // return wait time in us
}

/**
 * Read the predicted class (0-9).
 */
static inline int MLP_GetPrediction(MLPDev *dev)
{
    u32 status = XGpio_DiscreteRead(&dev->gpio_status, MLP_STATUS_CH);
    return (status >> MLP_STATUS_CLASS_POS) & 0x0F;
}

/**
 * Run complete inference on a single image.
 * Returns predicted class (0-9) or -1 on error.
 */
static inline int MLP_Run(MLPDev *dev, short *image_q88)
{
    MLP_SendImage(dev, image_q88);
    MLP_Start(dev);

    int ret = MLP_WaitDone(dev);
    if (ret < 0) return -1;  // timeout

    return MLP_GetPrediction(dev);
}

/**
 * Reset the MLP accelerator.
 */
static inline void MLP_Reset(MLPDev *dev)
{
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL,
                        MLP_CTRL_RST_MASK);
    usleep(10);
    XGpio_DiscreteWrite(&dev->gpio_ctrl, MLP_CTRL_CH_SIGNAL, 0);
    usleep(10);
}

/**
 * Get a human-readable status string.
 */
static inline const char* MLP_StatusStr(int status_code)
{
    switch (status_code) {
        case XST_SUCCESS:       return "OK";
        case XST_FAILURE:       return "FAIL";
        case XST_DEVICE_NOT_FOUND: return "NO_DEVICE";
        default:                return "UNKNOWN";
    }
}

#endif /* MLP_DRIVER_H */
