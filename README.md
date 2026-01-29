



```web
// API Call - background request, no navigation
const response = await fetch(IDV_SERVER + `/idv/login-ticket`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ login_ticket: login_ticket }),
  credentials: 'include'
});

// ❌ This would be a hyperlink (page navigation):
window.location.href = responseData.start_idv_uri;  
// → Browser navigates to new page

// ✅ This is an API call (background data exchange):
const response = await fetch(IDV_SERVER + `/idv/login-ticket`, {
  method: "POST",
  body: JSON.stringify({ login_ticket: login_ticket })
});
// → Sends data to server, gets response, stays on same page

// 1. API Call - consume login ticket (background)
await fetch('/idv/login-ticket', { method: 'POST', ... });
// → No page change, just authenticates

// 2. API Call - start verification (background)  
const response = await fetch('/idv/jp/cookie/start', { method: 'POST', ... });
const data = await response.json();

// 3. Hyperlink Navigation - redirect to verification
window.location.href = data.start_idv_uri;
// → Browser navigates to external verification site
```
