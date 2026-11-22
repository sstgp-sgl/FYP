// fetch wrappers: unified error shape {error:{code,message,details},request_id}.
function requestId() {
  return crypto.randomUUID ? crypto.randomUUID().replaceAll("-", "") : String(Date.now());
}

async function parse(response) {
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  const requestIdHeader = response.headers.get("X-Request-ID");
  if (!response.ok) {
    const err = body?.error || {
      code: "http_error",
      message: `HTTP ${response.status}`,
      details: null,
    };
    const error = new Error(err.message || `HTTP ${response.status}`);
    error.code = err.code;
    error.details = err.details;
    error.request_id = requestIdHeader || (body && body.request_id);
    error.status = response.status;
    throw error;
  }
  return body;
}

export async function getJSON(path) {
  const response = await fetch(path, {
    headers: { "X-Request-ID": requestId() },
  });
  return parse(response);
}

export async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Request-ID": requestId() },
    body: JSON.stringify(payload),
  });
  return parse(response);
}

export async function putJSON(path, payload) {
  const response = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json", "X-Request-ID": requestId() },
    body: JSON.stringify(payload),
  });
  return parse(response);
}

export async function uploadPLY(file, title) {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);
  const response = await fetch("/api/tour/scenes/import", {
    method: "POST",
    headers: { "X-Request-ID": requestId() },
    body: form,
  });
  return parse(response);
}

export async function fetchBinary(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`binary fetch failed: ${response.status}`);
  return response.arrayBuffer();
}

export function downloadUrl(storyboardId) {
  return `/api/tour/storyboards/${encodeURIComponent(storyboardId)}/download`;
}
