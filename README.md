

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
