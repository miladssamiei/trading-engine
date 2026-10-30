/*
 * bridge_client.h — DLL interface for MT5 EA to communicate with the bridge.
 *
 * Protocol: newline-delimited JSON over TCP to 127.0.0.1:<port>.
 *
 * Bar JSON (sent):  {"symbol":"EURUSD","open":1.1234,"high":...,"timestamp":...}\n
 * Response (recv):  {"action":"BUY","cairo_weight":0.85,...}\n
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * IMPORTANT — MQL5 callers MUST use BridgeSendReceiveRaw with CP_UTF8 byte
 * arrays.  MQL5's `string` -> `const char*` marshaling goes through the
 * system ANSI code page (CP_ACP / CP1252) which corrupts JSON containing
 * double-quote and brace characters.  The Raw variant accepts `unsigned char*`
 * (MQL5 `uchar[]`) and is encoding-transparent.
 *
 * See PROGRESS.md § "CP_ACP JSON corruption fix (2026-08-05)" for full analysis.
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * On x64, calling convention is __cdecl (the only convention on x64).
 */

#ifndef BRIDGE_CLIENT_H
#define BRIDGE_CLIENT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #define BRIDGE_API __declspec(dllexport)
#else
  #define BRIDGE_API
#endif

/*
 * Check if the bridge server is alive.
 *
 * Returns 1 if bridge responds with "pong\n", 0 otherwise.
 */
BRIDGE_API int BridgePing(int port);

/*
 * ── MQL5 primary export (USE THIS IN ALL MQL5 CODE) ──────────────────
 *
 * Send a bar JSON to the bridge and receive a signal JSON response.
 * Both buffers are encoding-transparent (`unsigned char*` = MQL5 `uchar[]`).
 * MQL5 callers MUST encode request bytes with  StringToCharArray(..., CP_UTF8)
 * and decode response bytes with   CharArrayToString(..., CP_UTF8).
 *
 *   port          — TCP port of the bridge server on 127.0.0.1
 *   request       — null-terminated UTF-8 JSON string (bar data)
 *   response      — caller-allocated buffer for the response JSON
 *   response_size — size of the response buffer in bytes
 *
 * Returns the number of bytes written to response (excluding null terminator),
 * or 0 on error. The response is always null-terminated if response_size > 0.
 */
BRIDGE_API int BridgeSendReceiveRaw(int port, const unsigned char *request,
                                    unsigned char *response, int response_size);

/* ═══════════════════════════════════════════════════════════════════════
 * DEPRECATED — DO NOT USE FROM MQL5
 *
 * BridgeSendReceive accepts `const char*` / `char*` buffers.
 * MQL5's `string`  -> `const char*` conversion goes through CP_ACP,
 * which CORRUPTS JSON.  This export exists only for non-MQL5 callers
 * (Python ctypes tests, C programs) where the caller controls encoding.
 * All MQL5 code MUST use BridgeSendReceiveRaw.
 * ═══════════════════════════════════════════════════════════════════════ */
BRIDGE_API int BridgeSendReceive(int port, const char *request,
                                 char *response, int response_size);

#ifdef __cplusplus
}
#endif

#endif /* BRIDGE_CLIENT_H */
