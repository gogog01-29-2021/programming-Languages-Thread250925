

```

Language	Interactive	Compile & Run
Haskell	stack ghci + main	stack build + ./exe
Python	python + main()	python main.py
C/C++	gdb ./program	gcc main.c && ./a.out
Rust	evcxr (3rd party)	cargo run

```

```build  ghci+main for cpp and c and rust and python
ghci+main for cpp and c and rust and python

# c & cpp
gdb ./myprogram     # Debugger - can call functions
lldb ./myprogram    # Alternative debugger

# Or compile and run:
gcc main.c -o main && ./main
g++ main.cpp -o main && ./main

# Rust
cargo run           # Compile and run (like stack build + run)
# No interactive interpreter built-in

# But there's:
evcxr               # Third-party Rust REPL

#Python
python              # Interactive interpreter (like ghci)
>>> main()          # Call functions
>>> import mymodule # Reload modules
>>> exit()

# Or run script:
python main.py      # Like stack build + run
```


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

```web design
Complete Flow Diagram                                                
                                                                       
  ┌────────────────────────────────────────────────────────────────────
  ─────────────┐                                                       
  │                              USER'S BROWSER                        
                │                                                      
  │                                                                    
                │                                                      
  │  1. User visits:                                                   
  https://idv-app.com/idv/jp?user_id=123&login_ticket=abc       │      
  │                                                                    
                │                                                      
  │  2. React loads LiquidIDVCookie.tsx                                
               │                                                       
  │     ┌──────────────────────────────────────────────────────────────
  ───────┐     │                                                       
  │     │  useEffect() runs automatically                              
          │     │                                                      
  │     │                                                              
          │     │                                                      
  │     │  Step 1: consumeLoginTicket()                                
          │     │                                                      
  │     │          ↓                                                   
          │     │                                                      
  │     │                                                              
  fetch("https://api.tomopayment.com/v1/idv/login-ticket")   │     │   
  │     │                                                              
          │     │                                                      
  └─────┼──────────────────────────┬───────────────────────────────────
  ────────┼─────┘                                                      
                                   │                                   
                                   │  POST { login_ticket: "abc" }     
                                   │  (sends data TO server)           
                                   ▼                                   
  ┌────────────────────────────────────────────────────────────────────
  ─────────────┐                                                       
  │                              IDV-SERVER (Haskell)                  
                │                                                      
  │                                                                    
                │                                                      
  │  LoginTicket.hs receives the request                               
               │                                                       
  │  ┌─────────────────────────────────────────────────────────────────
  ────────┐    │                                                       
  │  │  1. Check if login_ticket "abc" is valid in database            
          │    │                                                       
  │  │  2. If valid → generate access_ticket cookie                    
          │    │                                                       
  │  │  3. Return { status: "ok" } + Set-Cookie header                 
          │    │                                                       
  │  └─────────────────────────────────────────────────────────────────
  ────────┘    │                                                       
  │                                                                    
                │                                                      
  └────────────────────────────────────────────────────────────────────
  ─────────────┘                                                       
                                   │                                   
                                   │  Response: { status: "ok" }       
                                   │  + cookie saved in browser        
                                   ▼                                   
  ┌────────────────────────────────────────────────────────────────────
  ─────────────┐                                                       
  │                              USER'S BROWSER                        
                │                                                      
  │     ┌──────────────────────────────────────────────────────────────
  ───────┐     │                                                       
  │     │  Step 1 SUCCESS! ✓                                           
          │     │                                                      
  │     │                                                              
          │     │                                                      
  │     │  Step 2: handleStartVerification()                           
          │     │                                                      
  │     │          ↓                                                   
          │     │                                                      
  │     │                                                              
  fetch("https://api.tomopayment.com/v1/idv/jp/cookie/start")│     │   
  │     │                                                              
          │     │                                                      
  └─────┼──────────────────────────┬───────────────────────────────────
  ────────┼─────┘                                                      
                                   │                                   
                                   │  POST { user_id: "123", country:  
  "jp" }                                                               
                                   │  + cookie (automatic)             
                                   ▼                                   
  ┌────────────────────────────────────────────────────────────────────
  ─────────────┐                                                       
  │                              IDV-SERVER (Haskell)                  
                │                                                      
  │                                                                    
                │                                                      
  │  Start.hs receives the request                                     
               │                                                       
  │  ┌─────────────────────────────────────────────────────────────────
  ────────┐    │                                                       
  │  │  1. Verify cookie is valid                                      
          │    │                                                       
  │  │  2. Call Liquid (external company) API to create verification   
  session  │    │                                                      
  │  │  3. Get launch_url from Liquid                                  
          │    │                                                       
  │  │  4. Generate new URL:                                           
  https://idv-app.com/idv/jp/launch?token=xyz      │    │              
  │  │  5. Return { start_idv_uri:                                     
  "https://idv-app.com/idv/jp/launch?token=xyz" }│ │                   
  │  └─────────────────────────────────────────────────────────────────
  ────────┘    │                                                       
  │                                                                    
                │                                                      
  └────────────────────────────────────────────────────────────────────
  ─────────────┘                                                       
                                   │                                   
                                   │  Response: { start_idv_uri:       
  "https://..." }                                                      
                                   ▼                                   
  ┌────────────────────────────────────────────────────────────────────
  ─────────────┐                                                       
  │                              USER'S BROWSER                        
                │                                                      
  │     ┌──────────────────────────────────────────────────────────────
  ───────┐     │                                                       
  │     │  Step 2 SUCCESS! ✓                                           
          │     │                                                      
  │     │                                                              
          │     │                                                      
  │     │  Step 3: window.location.href = start_idv_uri                
          │     │                                                      
  │     │          ↓                                                   
          │     │                                                      
  │     │          Browser navigates to:                               
  https://idv-app.com/idv/jp/launch... │     │                         
  │     │          ↓                                                   
          │     │                                                      
  │     │          User sees Liquid's ID verification page             
          │     │                                                      
  │     │          ↓                                                   
          │     │                                                      
  │     │          User takes photo of ID, selfie, etc.                
          │     │                                                      
  │     │                                                              
          │     │                                                      
  │     └──────────────────────────────────────────────────────────────
  ───────┘     │                                                       
  │                                                                    
                │                                                      
  └────────────────────────────────────────────────────────────────────
  ─────────────┘                                                       
                                                                  
```

```API components + Data Flow
const response = await fetch(IDV_SERVER + `/idv/login-ticket`, {
//    │         │     │         │              │
//    │         │     │         │              └── API endpoint path
//    │         │     │         └── Base server URL (environment variable)
//    │         │     └── fetch() - browser's HTTP client function  
//    │         └── await - wait for async operation to complete
//    └── Store the response in 'response' variable

  method: "POST",
  //      │
  //      └── HTTP method - sending data TO server (not getting)

  headers: { 
    "Content-Type": "application/json" 
  },
  //│                    │
  //│                    └── Tell server: "I'm sending JSON data"
  //└── HTTP headers - metadata about the request

  body: JSON.stringify({ login_ticket: login_ticket }),
  //    │              │               │
  //    │              │               └── Variable containing the ticket value
  //    │              └── JavaScript object to send
  //    └── Convert object to JSON string for transmission

  credentials: 'include'
  //           │
  //           └── Include cookies/session data in request
});

// 1. JavaScript constructs URL:
const fullURL = IDV_SERVER + `/idv/login-ticket`
// Result: "https://api.myserver.com/idv/login-ticket"

// 2. Browser makes HTTP request (BEHIND THE SCENES):
POST https://api.myserver.com/idv/login-ticket
Content-Type: application/json
Cookie: session_id=abc123; auth_token=xyz789

{"login_ticket": "ticket_12345"}

// 3. Server processes and responds:
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "ok", "message": "Ticket consumed successfully"}

// 4. JavaScript receives response:
const data = await response.json();
console.log(data.status); // "ok"

// Your React component is running on:
// https://myapp.com/verification-page

// 1. API Call happens in background:
await fetch("https://api.server.com/idv/login-ticket", {
  method: "POST",
  body: JSON.stringify({ login_ticket: "abc123" })
});
// → No URL change, user still sees: https://myapp.com/verification-page

// 2. After API success, THEN navigation happens:
window.location.href = responseData.start_idv_uri;
// → NOW browser navigates to: https://external-idv.com/verify/xyz789
```
