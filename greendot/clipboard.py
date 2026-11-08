"""Clipboard: text (any OS) and images (Windows). OWNER: Markus

Text uses pyperclip. Images go straight to the Windows clipboard as a DIB
(a BMP file minus its 14-byte header), the format Word, PowerPoint, OneNote,
Paint and chat apps all accept on paste.
"""

import sys

import cv2


def copy_text(text):
    import pyperclip
    pyperclip.copy(text)


def copy_image(img):
    """Put a BGR image on the clipboard. Raises on failure or on non-Windows."""
    if sys.platform != "win32":
        raise NotImplementedError("copying images is only supported on Windows")
    import ctypes
    from ctypes import wintypes

    ok, bmp = cv2.imencode(".bmp", img)
    if not ok:
        raise ValueError("could not encode the image")
    dib = bmp.tobytes()[14:]   # drop BITMAPFILEHEADER; the rest is a DIB

    CF_DIB, GMEM_MOVEABLE = 8, 0x0002
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    # 64-bit handles must not be truncated to 32-bit ints.
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
    if not handle:
        raise MemoryError("GlobalAlloc failed")
    ctypes.memmove(kernel32.GlobalLock(handle), dib, len(dib))
    kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        kernel32.GlobalFree(handle)
        raise RuntimeError("clipboard is busy")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_DIB, handle):
            kernel32.GlobalFree(handle)
            raise RuntimeError("SetClipboardData failed")
        # On success Windows owns the memory; it must not be freed here.
    finally:
        user32.CloseClipboard()
