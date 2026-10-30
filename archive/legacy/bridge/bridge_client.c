/*
 * bridge_client.c — Winsock2 TCP client DLL for MT5 EA bridge communication.
 *
 * Compile (MinGW-w64 x64):
 *   gcc -shared -O2 -o bridge_client.dll bridge_client.c -lws2_32
 */

#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#include <string.h>

#include "bridge_client.h"

#define RECV_CHUNK 4096

static int _wsa_ready = 0;

static int _init_wsa(void)
{
    if (_wsa_ready)
        return 1;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        return 0;
    _wsa_ready = 1;
    return 1;
}

static SOCKET _connect(int port)
{
    SOCKET sock = INVALID_SOCKET;
    struct sockaddr_in addr;

    if (!_init_wsa())
        return INVALID_SOCKET;

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET)
        return INVALID_SOCKET;

    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((unsigned short)port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sock);
        return INVALID_SOCKET;
    }

    return sock;
}

static int _send_all(SOCKET sock, const char *data, int len)
{
    int sent = 0;
    while (sent < len) {
        int n = send(sock, data + sent, len - sent, 0);
        if (n <= 0)
            return 0;
        sent += n;
    }
    return 1;
}

static int _recv_line(SOCKET sock, char *buf, int buf_size)
{
    int total = 0;
    char tmp[RECV_CHUNK];
    int pos;

    while (total < buf_size - 1) {
        int n = recv(sock, tmp, RECV_CHUNK, 0);
        if (n <= 0)
            break;
        for (int i = 0; i < n; i++) {
            if (tmp[i] == '\n') {
                buf[total] = '\0';
                return total;
            }
            if (total < buf_size - 1) {
                buf[total++] = tmp[i];
            }
        }
    }
    if (total > 0)
        buf[total] = '\0';
    return total;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

BRIDGE_API int BridgePing(int port)
{
    SOCKET sock = _connect(port);
    if (sock == INVALID_SOCKET)
        return 0;

    const char *ping = "{\"ping\":1}\n";
    if (!_send_all(sock, ping, (int)strlen(ping))) {
        closesocket(sock);
        return 0;
    }

    char buf[256];
    int  n = _recv_line(sock, buf, sizeof(buf));
    closesocket(sock);

    if (n > 0 && strstr(buf, "pong") != NULL)
        return 1;
    return 0;
}

BRIDGE_API int BridgeSendReceive(int port, const char *request,
                                 char *response, int response_size)
{
    if (!request || !response || response_size <= 0)
        return 0;

    SOCKET sock = _connect(port);
    if (sock == INVALID_SOCKET)
        return 0;

    /* Append newline if not present */
    int req_len = (int)strlen(request);
    {
        char *full_req = (char *)_alloca(req_len + 2);
        memcpy(full_req, request, req_len);
        if (req_len == 0 || full_req[req_len - 1] != '\n') {
            full_req[req_len]     = '\n';
            full_req[req_len + 1] = '\0';
            req_len++;
        }
        if (!_send_all(sock, full_req, req_len)) {
            closesocket(sock);
            return 0;
        }
    }

    int n = _recv_line(sock, response, response_size);
    closesocket(sock);

    return n;
}

BRIDGE_API int BridgeSendReceiveRaw(int port, const unsigned char *request,
                                    unsigned char *response, int response_size)
{
    return BridgeSendReceive(port, (const char *)request,
                             (char *)response, response_size);
}
