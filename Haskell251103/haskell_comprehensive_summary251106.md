# Learn You a Haskell for Great Good - Comprehensive Summary

## Book Overview
This is a beginner-friendly Haskell book covering fundamentals through advanced concepts like Monads, Functors, and Zippers. It's designed for programmers coming from imperative languages.

---

## PART 1: CORE FUNDAMENTALS

### Chapter 1: Starting Out

**Key Concepts:**
- **Function Calls**: Prefix notation (function comes first), infix operators
- **Basic Operations**: Arithmetic, Boolean algebra, comparison
- **Function Application**: Highest precedence in Haskell

**Key Code Examples:**
```haskell
-- Simple function definition
doubleMe x = x + x

-- Multiple parameters
doubleUs x y = x * 2 + y * 2

-- Using other functions
doubleUs x y = doubleMe x + doubleMe y

-- If expressions (else is mandatory!)
doubleSmallNumber x = if x > 100
                       then x
                       else x*2

-- Backticks for infix notation
92 `div` 10  -- => 9
```

**Lists:**
```haskell
-- List concatenation
[1,2,3] ++ [4,5,6]  -- => [1,2,3,4,5,6]

-- Cons operator
1:[2,3,4]  -- => [1,2,3,4]

-- List indexing
[1,2,3,4] !! 2  -- => 3

-- List operations
head [1,2,3,4]    -- => 1
tail [1,2,3,4]    -- => [2,3,4]
last [1,2,3,4]    -- => 4
init [1,2,3,4]    -- => [1,2,3]
length [1,2,3,4]  -- => 4
null []           -- => True
reverse [1,2,3]   -- => [3,2,1]
```

**Ranges:**
```haskell
[1..10]        -- => [1,2,3,4,5,6,7,8,9,10]
[2,4..20]      -- => [2,4,6,8,10,12,14,16,18,20]
['a'..'z']     -- => "abcdefghijklmnopqrstuvwxyz"

-- Infinite lists (lazy evaluation!)
[1..]          -- Infinite list of natural numbers
take 10 [1..]  -- => [1,2,3,4,5,6,7,8,9,10]
```

**List Comprehensions:**
```haskell
-- Basic comprehension
[x*2 | x <- [1..10]]  -- => [2,4,6,8,10,12,14,16,18,20]

-- With predicate (filtering)
[x*2 | x <- [1..10], x*2 >= 12]  -- => [12,14,16,18,20]

-- Multiple predicates
[x | x <- [10..20], x /= 13, x /= 15, x /= 19]

-- Multiple lists
[x*y | x <- [2,5,10], y <- [8,10,11]]

-- With functions
length' xs = sum [1 | _ <- xs]
```

**Tuples:**
```haskell
-- Pairs
(1, 2)
("hello", True)

-- Tuple functions
fst (8, 11)  -- => 8
snd (8, 11)  -- => 11

-- zip function
zip [1,2,3] [4,5,6]  -- => [(1,4), (2,5), (3,6)]
```

---

### Chapter 2: Believe the Type

**Type System Basics:**
```haskell
-- Explicit type declarations
removeNonUppercase :: String -> String
removeNonUppercase st = [c | c <- st, c `elem` ['A'..'Z']]

addThree :: Int -> Int -> Int -> Int
addThree x y z = x + y + z
```

**Common Types:**
- `Int`: Bounded integers
- `Integer`: Unbounded integers
- `Float`: Single-precision floating point
- `Double`: Double-precision floating point
- `Bool`: Boolean values (True/False)
- `Char`: Character
- `String`: List of characters ([Char])

**Type Variables (Polymorphism):**
```haskell
head :: [a] -> a
fst :: (a, b) -> a
```

**Type Classes:**
```haskell
-- Eq - for equality testing
(==) :: Eq a => a -> a -> Bool

-- Ord - for ordering
(>) :: Ord a => a -> a -> Bool

-- Show - can be presented as strings
show :: Show a => a -> String
show 3  -- => "3"

-- Read - opposite of Show
read :: Read a => String -> a
read "5" :: Int  -- => 5

-- Num - numeric types
(+) :: Num a => a -> a -> a

-- Integral - Int and Integer
fromIntegral :: (Integral a, Num b) => a -> b
```

---

### Chapter 3: Syntax in Functions

**Pattern Matching:**
```haskell
-- Basic pattern matching
lucky :: Int -> String
lucky 7 = "LUCKY NUMBER SEVEN!"
lucky x = "Sorry, you're out of luck, pal!"

-- Recursive patterns
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Tuple patterns
addVectors :: (Double, Double) -> (Double, Double) -> (Double, Double)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

-- List patterns
head' :: [a] -> a
head' [] = error "Can't call head on empty list"
head' (x:_) = x

-- As-patterns (keeping reference to whole)
firstLetter :: String -> String
firstLetter "" = "Empty string!"
firstLetter all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
```

**Guards:**
```haskell
bmiTell :: Double -> String
bmiTell bmi
    | bmi <= 18.5 = "Underweight"
    | bmi <= 25.0 = "Normal"
    | bmi <= 30.0 = "Overweight"
    | otherwise   = "Obese"

-- Guards with multiple parameters
max' :: Ord a => a -> a -> a
max' a b
    | a <= b    = b
    | otherwise = a
```

**Where Bindings:**
```haskell
bmiTell :: Double -> Double -> String
bmiTell weight height
    | bmi <= skinny = "Underweight"
    | bmi <= normal = "Normal"
    | bmi <= fat    = "Overweight"
    | otherwise     = "Obese"
    where bmi = weight / height ^ 2
          skinny = 18.5
          normal = 25.0
          fat = 30.0

-- Pattern matching in where
where bmi = weight / height ^ 2
      (skinny, normal, fat) = (18.5, 25.0, 30.0)
```

**Let Bindings:**
```haskell
-- let in expressions
cylinder :: Double -> Double -> Double
cylinder r h =
    let sideArea = 2 * pi * r * h
        topArea = pi * r ^ 2
    in sideArea + 2 * topArea

-- let in list comprehensions
[bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi > 25.0]
```

**Case Expressions:**
```haskell
head' :: [a] -> a
head' xs = case xs of []    -> error "No head for empty lists!"
                       (x:_) -> x

describeList :: [a] -> String
describeList ls = "The list is " ++ case ls of []  -> "empty."
                                                 [x] -> "a singleton list."
                                                 xs  -> "a longer list."
```

---

### Chapter 4: Hello Recursion!

**Recursion Examples:**
```haskell
-- Maximum of a list
maximum' :: Ord a => [a] -> a
maximum' [] = error "maximum of empty list!"
maximum' [x] = x
maximum' (x:xs) = max x (maximum' xs)

-- Replicate
replicate' :: Int -> a -> [a]
replicate' n x
    | n <= 0    = []
    | otherwise = x : replicate' (n-1) x

-- Take
take' :: Int -> [a] -> [a]
take' n _
    | n <= 0   = []
take' _ []     = []
take' n (x:xs) = x : take' (n-1) xs

-- Reverse
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse' xs ++ [x]

-- Repeat (infinite list)
repeat' :: a -> [a]
repeat' x = x : repeat' x

-- Zip
zip' :: [a] -> [b] -> [(a,b)]
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x,y) : zip' xs ys

-- QuickSort
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
    let smallerOrEqual = [a | a <- xs, a <= x]
        larger = [a | a <- xs, a > x]
    in quicksort smallerOrEqual ++ [x] ++ quicksort larger
```

---

## PART 2: HIGHER-ORDER CONCEPTS

### Chapter 5: Higher-Order Functions

**Currying:**
Every function in Haskell officially takes only one parameter. Multi-parameter functions are actually curried functions returning functions.

```haskell
-- These are equivalent
max 4 5
(max 4) 5

-- Partial application
multThree :: Int -> Int -> Int -> Int
multThree x y z = x * y * z

multTwoWithNine = multThree 9
-- multTwoWithNine is now a function: Int -> Int -> Int
```

**Sections (infix partial application):**
```haskell
divideByTen :: Double -> Double
divideByTen = (/10)

isUpperAlphanum :: Char -> Bool
isUpperAlphanum = (`elem` ['A'..'Z'])
```

**Higher-Order Functions:**
```haskell
-- Function that takes a function
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

applyTwice (+3) 10    -- => 16
applyTwice (++ " HAHA") "HEY"  -- => "HEY HAHA HAHA"

-- zipWith
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

zipWith' (+) [1,2,3] [4,5,6]  -- => [5,7,9]
zipWith' max [1,2] [4,1]      -- => [4,2]

-- flip
flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f y x = f x y

flip' zip [1,2,3] "hello"  -- => [('h',1),('e',2),('l',3)]
```

**Map and Filter:**
```haskell
-- map
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs

map (+3) [1,5,3,1,6]      -- => [4,8,6,4,9]
map (replicate 3) [3..6]  -- => [[3,3,3],[4,4,4],[5,5,5],[6,6,6]]

-- filter
filter :: (a -> Bool) -> [a] -> [a]
filter _ [] = []
filter p (x:xs)
    | p x       = x : filter p xs
    | otherwise = filter p xs

filter (>3) [1,5,3,2,1,6,4,3,2,1]  -- => [5,6,4]
filter even [1..10]                 -- => [2,4,6,8,10]
```

**Lambdas:**
```haskell
-- Anonymous functions
(\x -> x + 3)

map (\x -> x + 3) [1,2,3]  -- => [4,5,6]

-- Multiple parameters
zipWith (\a b -> (a * 30 + 3) / b) [5,4,3,2,1] [1,2,3,4,5]

-- Pattern matching in lambdas
map (\(a,b) -> a + b) [(1,2),(3,5),(6,3)]
```

**Folds:**
```haskell
-- foldl (left fold)
sum' :: Num a => [a] -> a
sum' xs = foldl (\acc x -> acc + x) 0 xs
-- Even simpler:
sum' = foldl (+) 0

-- foldr (right fold)
map' :: (a -> b) -> [a] -> [b]
map' f xs = foldr (\x acc -> f x : acc) [] xs

-- foldl1 and foldr1 (assume non-empty list)
maximum' :: Ord a => [a] -> a
maximum' = foldl1 max

-- Fold examples
reverse' :: [a] -> [a]
reverse' = foldl (\acc x -> x : acc) []

product' :: Num a => [a] -> a
product' = foldl (*) 1

filter' :: (a -> Bool) -> [a] -> [a]
filter' p = foldr (\x acc -> if p x then x : acc else acc) []
```

**Scans (like folds but keep intermediate results):**
```haskell
scanl (+) 0 [3,5,2,1]  -- => [0,3,8,10,11]
scanr (+) 0 [3,5,2,1]  -- => [11,8,3,1,0]
```

**Function Application and Composition:**
```haskell
-- $ operator (function application)
sum (map sqrt [1..130])
sum $ map sqrt [1..130]  -- Same thing

-- . operator (function composition)
map (\x -> negate (abs x)) [5,-3,-6,7]
map (negate . abs) [5,-3,-6,7]  -- => [-5,-3,-6,-7]

-- Point-free style (omitting parameters)
sum' :: Num a => [a] -> a
sum' xs = foldl (+) 0 xs
-- Point-free:
sum' = foldl (+) 0

fn = ceiling . negate . tan . cos . max 50
```

---

### Chapter 6: Modules

**Importing Modules:**
```haskell
import Data.List
import Data.List (nub, sort)  -- Only specific functions
import Data.List hiding (nub) -- All except nub
import qualified Data.Map as M -- Qualified import
```

**Useful Module Functions:**
```haskell
-- Data.List
intersperse '.' "MONKEY"  -- => "M.O.N.K.E.Y"
intercalate " " ["hey","there"]  -- => "hey there"
transpose [[1,2,3],[4,5,6]]  -- => [[1,4],[2,5],[3,6]]
foldl' (+) 0 [1..1000000]  -- Strict left fold
concat [[3,4,5],[2,3,4]]  -- => [3,4,5,2,3,4]
concatMap (replicate 4) [1..3]  -- => [1,1,1,1,2,2,2,2,3,3,3,3]
and [True,True,False]  -- => False
any (==4) [2,3,5,6]  -- => False
all (>4) [6,9,10]  -- => True
iterate (*2) 1  -- => [1,2,4,8,16,32,...]
splitAt 3 "heyman"  -- => ("hey","man")
takeWhile (<3) [1,2,3,4]  -- => [1,2]
dropWhile (<3) [1,2,3,4]  -- => [3,4]
span (<3) [1,2,3,4,5]  -- => ([1,2],[3,4,5])
break (>3) [1,2,3,4,5]  -- => ([1,2,3],[4,5])
group "aaabbbbcca"  -- => ["aaa","bbbb","cc","a"]
inits "wow"  -- => ["","w","wo","wow"]
tails "wow"  -- => ["wow","ow","w",""]
isInfixOf "cat" "im a cat burglar"  -- => True
isPrefixOf "hey" "hey there!"  -- => True
elem 4 [1,2,3,4]  -- => True
find (>4) [1,2,3,4,5,6]  -- => Just 5
lines "first line\nsecond line"  -- => ["first line","second line"]
delete 'h' "hey there"  -- => "ey there"
[1..10] \\ [2,5,9]  -- => [1,3,4,6,7,8,10] (list difference)
union [1,2,3] [2,3,4]  -- => [1,2,3,4]
intersect [1,2,3] [2,3,4]  -- => [2,3]
insert 4 [1,2,3,5,6]  -- => [1,2,3,4,5,6]

-- Data.Char
ord 'a'  -- => 97
chr 97   -- => 'a'
digitToInt '9'  -- => 9
isControl '\n'  -- => True
isSpace ' '  -- => True
isLower 'a'  -- => True
isUpper 'A'  -- => True
isAlpha 'a'  -- => True
isAlphaNum 'a'  -- => True
isDigit '9'  -- => True
toUpper 'a'  -- => 'A'
toLower 'A'  -- => 'a'

-- Data.Map
import qualified Data.Map as Map

-- Creating maps
Map.fromList [("MS",1),("MS",2),("MS",3)]  -- => fromList [("MS",3)]
Map.lookup "betty" phoneBook  -- => Just "555-2938"
Map.insert "grace" "341-9021" phoneBook
Map.size phoneBook
Map.map (*100) (Map.fromList [(1,1),(2,4),(3,9)])
```

**Making Your Own Modules:**
```haskell
-- Geometry.hs
module Geometry
( sphereVolume
, sphereArea
, cubeVolume
, cubeArea
) where

sphereVolume :: Float -> Float
sphereVolume radius = (4.0 / 3.0) * pi * (radius ^ 3)

sphereArea :: Float -> Float
sphereArea radius = 4 * pi * (radius ^ 2)

-- Hierarchical modules
-- Geometry/Sphere.hs
module Geometry.Sphere
( volume
, area
) where

volume :: Float -> Float
volume radius = (4.0 / 3.0) * pi * (radius ^ 3)
```

---

## PART 3: ADVANCED TYPE SYSTEM

### Chapter 7: Making Our Own Types and Type Classes

**Defining Data Types:**
```haskell
-- Basic algebraic data type
data Bool = False | True

-- With parameters
data Shape = Circle Float Float Float 
           | Rectangle Float Float Float Float

-- Using it
surface :: Shape -> Float
surface (Circle _ _ r) = pi * r ^ 2
surface (Rectangle x1 y1 x2 y2) = abs (x2 - x1) * abs (y2 - y1)
```

**Record Syntax:**
```haskell
data Person = Person { firstName :: String
                     , lastName :: String
                     , age :: Int
                     } deriving (Show)

-- Creating instances
Person {firstName="John", lastName="Doe", age=25}

-- Accessing fields
firstName john  -- => "John"
```

**Type Parameters:**
```haskell
-- Parameterized type
data Maybe a = Nothing | Just a

-- Examples
Just 3 :: Maybe Int
Nothing :: Maybe a

-- Type constructor (Maybe) vs Value constructor (Just, Nothing)
data Vector a = Vector a a a deriving (Show)

vplus :: Num a => Vector a -> Vector a -> Vector a
(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)
```

**Derived Instances:**
```haskell
data Person = Person { firstName :: String
                     , lastName :: String
                     , age :: Int
                     } deriving (Eq, Show, Read, Ord)

-- Now we can:
Person {firstName="John", lastName="Doe", age=25} == 
Person {firstName="John", lastName="Doe", age=25}  -- => True

read "Person {firstName=\"John\", lastName=\"Doe\", age=25}" :: Person
```

**Type Synonyms:**
```haskell
type String = [Char]
type PhoneNumber = String
type Name = String
type PhoneBook = [(Name, PhoneNumber)]

-- Parameterized type synonyms
type AssocList k v = [(k,v)]
```

**Recursive Data Structures:**
```haskell
-- Our own list
data List a = Empty | Cons a (List a) deriving (Show, Read, Eq, Ord)

-- Binary search tree
data Tree a = EmptyTree | Node a (Tree a) (Tree a) deriving (Show)

singleton :: a -> Tree a
singleton x = Node x EmptyTree EmptyTree

treeInsert :: Ord a => a -> Tree a -> Tree a
treeInsert x EmptyTree = singleton x
treeInsert x (Node a left right)
    | x == a = Node x left right
    | x < a  = Node a (treeInsert x left) right
    | x > a  = Node a left (treeInsert x right)

treeElem :: Ord a => a -> Tree a -> Bool
treeElem x EmptyTree = False
treeElem x (Node a left right)
    | x == a = True
    | x < a  = treeElem x left
    | x > a  = treeElem x right
```

**Type Classes 102:**
```haskell
-- Defining a type class
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x == y = not (x /= y)
    x /= y = not (x == y)

-- Making types instances of type classes
data TrafficLight = Red | Yellow | Green

instance Eq TrafficLight where
    Red == Red = True
    Green == Green = True
    Yellow == Yellow = True
    _ == _ = False

instance Show TrafficLight where
    show Red = "Red light"
    show Yellow = "Yellow light"
    show Green = "Green light"

-- Subclassing
class Eq a => Ord a where
    compare :: a -> a -> Ordering
    (<), (<=), (>=), (>) :: a -> a -> Bool
    max, min :: a -> a -> a

-- Parameterized types as instances
instance Eq a => Eq (Maybe a) where
    Just x == Just y = x == y
    Nothing == Nothing = True
    _ == _ = False
```

**Yes-No Type Class (Custom):**
```haskell
class YesNo a where
    yesno :: a -> Bool

instance YesNo Int where
    yesno 0 = False
    yesno _ = True

instance YesNo [a] where
    yesno [] = False
    yesno _ = True

instance YesNo Bool where
    yesno = id

instance YesNo (Maybe a) where
    yesno (Just _) = True
    yesno Nothing = False

yesnoIf :: YesNo y => y -> a -> a -> a
yesnoIf yesnoVal yesResult noResult =
    if yesno yesnoVal then yesResult else noResult
```

**The Functor Type Class:**
```haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- Maybe as a Functor
instance Functor Maybe where
    fmap f (Just x) = Just (f x)
    fmap f Nothing = Nothing

fmap (+3) (Just 3)  -- => Just 6
fmap (*2) Nothing   -- => Nothing

-- Tree as a Functor
instance Functor Tree where
    fmap f EmptyTree = EmptyTree
    fmap f (Node x left right) = Node (f x) (fmap f left) (fmap f right)

-- Either as a Functor
instance Functor (Either a) where
    fmap f (Right x) = Right (f x)
    fmap f (Left x) = Left x
```

**Kinds:**
```haskell
-- Kinds are types of types
:k Int            -- Int :: *
:k Maybe          -- Maybe :: * -> *
:k Maybe Int      -- Maybe Int :: *
:k Either         -- Either :: * -> * -> *
:k Either String  -- Either String :: * -> *
```

---

## PART 4: INPUT/OUTPUT AND MONADS

### Chapter 8 & 9: Input and Output

**Basic I/O:**
```haskell
-- Hello World
main = putStrLn "Hello, world!"

-- Gluing I/O actions
main = do
    putStrLn "Hello, what's your name?"
    name <- getLine
    putStrLn ("Hey " ++ name ++ ", you rock!")

-- let in do blocks (no 'in' part!)
import Data.Char
main = do
    putStrLn "What's your first name?"
    firstName <- getLine
    let bigFirstName = map toUpper firstName
    putStrLn ("Hey " ++ bigFirstName ++ "!")
```

**Useful I/O Functions:**
```haskell
-- putStr (no newline)
main = do
    putStr "Hey "
    putStr "there!"

-- putChar
main = do
    putChar 'h'
    putChar 'e'
    putChar 'y'

-- print (show then putStrLn)
main = do
    print True
    print 2
    print "haha"

-- when (Control.Monad)
import Control.Monad
main = do
    input <- getLine
    when (input == "SWORDFISH") $ do
        putStrLn input

-- sequence
main = do
    rs <- sequence [getLine, getLine, getLine]
    print rs

-- mapM and mapM_
mapM print [1,2,3]
mapM_ print [1,2,3]  -- Discards results

-- forever (infinite loop)
import Control.Monad
import Data.Char
main = forever $ do
    putStr "Give me some input: "
    l <- getLine
    putStrLn $ map toUpper l

-- forM (like mapM but parameters flipped)
import Control.Monad
main = do
    colors <- forM [1,2,3,4] $ \a -> do
        putStrLn $ "Which color do you associate with " ++ show a ++ "?"
        getLine
    putStrLn "The colors that you associate with 1, 2, 3 and 4 are:"
    mapM putStrLn colors
```

**File I/O:**
```haskell
-- Reading files
import System.IO
main = do
    handle <- openFile "girlfriend.txt" ReadMode
    contents <- hGetContents handle
    putStr contents
    hClose handle

-- Using withFile (automatic cleanup)
import System.IO
main = do
    withFile "girlfriend.txt" ReadMode $ \handle -> do
        contents <- hGetContents handle
        putStr contents

-- readFile (simpler)
import System.IO
main = do
    contents <- readFile "girlfriend.txt"
    putStr contents

-- writeFile
import System.IO
main = do
    contents <- readFile "girlfriend.txt"
    writeFile "girlfriendcaps.txt" (map toUpper contents)

-- appendFile
main = do
    appendFile "todo.txt" "Iron the dishes\n"
```

**Command-Line Arguments:**
```haskell
import System.Environment
import Data.List

main = do
    args <- getArgs
    progName <- getProgName
    putStrLn "The arguments are:"
    mapM putStrLn args
    putStrLn "The program name is:"
    putStrLn progName
```

**Randomness:**
```haskell
import System.Random

-- random
random (mkStdGen 100) :: (Int, StdGen)

-- randoms (infinite list)
take 5 $ randoms (mkStdGen 11) :: [Int]

-- randomR (in range)
randomR (1,6) (mkStdGen 359353)  -- => (3, StdGen ...)

-- randomRs (infinite list in range)
take 10 $ randomRs ('a','z') (mkStdGen 3)

-- I/O randomness
import System.Random
main = do
    gen <- getStdGen
    putStrLn $ take 20 (randomRs ('a','z') gen)
```

**Bytestrings:**
```haskell
-- Strict bytestrings
import qualified Data.ByteString as S
import qualified Data.ByteString.Char8 as S8

-- Lazy bytestrings
import qualified Data.ByteString.Lazy as L
import qualified Data.ByteString.Lazy.Char8 as L8

-- Efficient file copying
import System.Environment
import qualified Data.ByteString.Lazy as B

main = do
    (fileName1:fileName2:_) <- getArgs
    copyFile fileName1 fileName2

copyFile :: FilePath -> FilePath -> IO ()
copyFile source dest = do
    contents <- B.readFile source
    B.writeFile dest contents
```

---

## PART 5: SOLVING PROBLEMS FUNCTIONALLY

### Chapter 10: Functionally Solving Problems

**RPN Calculator:**
```haskell
import Data.List

solveRPN :: String -> Double
solveRPN = head . foldl foldingFunction [] . words
    where foldingFunction (x:y:ys) "*" = (x * y):ys
          foldingFunction (x:y:ys) "+" = (x + y):ys
          foldingFunction (x:y:ys) "-" = (y - x):ys
          foldingFunction (x:y:ys) "/" = (y / x):ys
          foldingFunction (x:y:ys) "^" = (y ** x):ys
          foldingFunction (x:xs) "ln" = log x:xs
          foldingFunction xs "sum" = [sum xs]
          foldingFunction xs numberString = read numberString:xs

-- Usage:
solveRPN "10 4 3 + 2 * -"  -- => -4.0
solveRPN "2.7 ln"            -- => 0.993...
```

**Path Finding (Heathrow to London):**
```haskell
data Section = Section { getA :: Int, getB :: Int, getC :: Int }
               deriving (Show)
type RoadSystem = [Section]
data Label = A | B | C deriving (Show)
type Path = [(Label, Int)]

roadStep :: (Path, Path) -> Section -> (Path, Path)
roadStep (pathA, pathB) (Section a b c) =
    let timeA = sum (map snd pathA)
        timeB = sum (map snd pathB)
        forwardTimeToA = timeA + a
        crossTimeToA = timeB + b + c
        forwardTimeToB = timeB + b
        crossTimeToB = timeA + a + c
        newPathToA = if forwardTimeToA <= crossTimeToA
                     then (A,a):pathA
                     else (C,c):(B,b):pathB
        newPathToB = if forwardTimeToB <= crossTimeToB
                     then (B,b):pathB
                     else (C,c):(A,a):pathA
    in (newPathToA, newPathToB)

optimalPath :: RoadSystem -> Path
optimalPath roadSystem =
    let (bestAPath, bestBPath) = foldl roadStep ([],[]) roadSystem
    in if sum (map snd bestAPath) <= sum (map snd bestBPath)
       then reverse bestAPath
       else reverse bestBPath
```

---

## PART 6: FUNCTORS, APPLICATIVES, MONOIDS, MONADS

### Chapter 11: Applicative Functors

**Functors Redux:**
```haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- IO as a Functor
instance Functor IO where
    fmap f action = do
        result <- action
        return (f result)

main = do
    line <- fmap reverse getLine
    putStrLn $ "You said " ++ line ++ " backwards!"

-- Functions as Functors
instance Functor ((->) r) where
    fmap = (.)

-- fmap (+3) (*2) 5 is the same as (+3) . (*2) $ 5
```

**Functor Laws:**
```haskell
-- Law 1: fmap id = id
fmap id (Just 3)  -- => Just 3

-- Law 2: fmap (f . g) = fmap f . fmap g
fmap ((+1) . (*2)) (Just 3)  -- => Just 7
fmap (+1) . fmap (*2) $ Just 3  -- => Just 7
```

**Applicative Functors:**
```haskell
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

-- Maybe as Applicative
instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> something = fmap f something

-- Usage
pure (+3) <*> Just 9  -- => Just 12
Just (+3) <*> Just 9  -- => Just 12
Just (++"hahah") <*> Nothing  -- => Nothing

-- Applicative style
pure (+) <*> Just 3 <*> Just 5  -- => Just 8
(++) <$> Just "johntra" <*> Just "volta"  -- => Just "johntravolta"

-- <$> is fmap as an infix operator
fmap (+3) (Just 4)  -- => Just 7
(+3) <$> Just 4     -- => Just 7
```

**Lists as Applicatives:**
```haskell
instance Applicative [] where
    pure x = [x]
    fs <*> xs = [f x | f <- fs, x <- xs]

-- Examples
[(*2),(+3)] <*> [1,2,3]  -- => [2,4,6,4,5,6]
[(+),(*)] <*> [1,2] <*> [3,4]  -- => [4,5,5,6,3,4,6,8]

-- Non-deterministic computations
[x*y | x <- [2,5,10], y <- [8,10,11]]
(*) <$> [2,5,10] <*> [8,10,11]  -- Same thing!
```

**IO as Applicative:**
```haskell
instance Applicative IO where
    pure = return
    a <*> b = do
        f <- a
        x <- b
        return (f x)

-- Example
main = do
    a <- (++) <$> getLine <*> getLine
    putStrLn $ "The two lines concatenated: " ++ a
```

**Functions as Applicatives:**
```haskell
instance Applicative ((->) r) where
    pure x = \_ -> x
    f <*> g = \x -> f x (g x)

-- Usage
(+) <$> (+3) <*> (*100) $ 5  -- => 508
-- ((+3) 5) + ((*100) 5) = 8 + 500 = 508
```

**ZipList:**
```haskell
instance Applicative ZipList where
    pure x = ZipList (repeat x)
    ZipList fs <*> ZipList xs = ZipList (zipWith (\f x -> f x) fs xs)

-- Usage
getZipList $ (+) <$> ZipList [1,2,3] <*> ZipList [100,100,100]
-- => [101,102,103]
```

**Useful Applicative Functions:**
```haskell
-- liftA2
liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 f a b = f <$> a <*> b

liftA2 (:) (Just 3) (Just [4])  -- => Just [3,4]

-- sequenceA
sequenceA :: Applicative f => [f a] -> f [a]
sequenceA = foldr (liftA2 (:)) (pure [])

sequenceA [Just 3, Just 2, Just 1]  -- => Just [3,2,1]
sequenceA [[1,2],[3,4]]  -- => [[1,3],[1,4],[2,3],[2,4]]
```

---

### Chapter 12: Monoids

**newtype:**
```haskell
newtype ZipList a = ZipList { getZipList :: [a] }

-- Difference from data:
-- - newtype is faster (no wrapping/unwrapping overhead)
-- - newtype can only have one value constructor with one field
-- - newtype is lazy differently than data

-- type vs newtype vs data:
-- type: just a synonym, same as original type
-- newtype: separate type, one constructor, one field
-- data: separate type, multiple constructors possible
```

**The Monoid Type Class:**
```haskell
class Monoid m where
    mempty :: m
    mappend :: m -> m -> m
    mconcat :: [m] -> m
    mconcat = foldr mappend mempty

-- Monoid laws:
-- mempty `mappend` x = x
-- x `mappend` mempty = x
-- (x `mappend` y) `mappend` z = x `mappend` (y `mappend` z)
```

**Lists as Monoids:**
```haskell
instance Monoid [a] where
    mempty = []
    mappend = (++)

[1,2,3] `mappend` [4,5,6]  -- => [1,2,3,4,5,6]
mconcat [[1,2],[3,4],[5,6]]  -- => [1,2,3,4,5,6]
```

**Product and Sum:**
```haskell
-- Since numbers can be monoids in multiple ways, we use newtype wrappers

-- Product (multiplication)
newtype Product a = Product { getProduct :: a }
instance Num a => Monoid (Product a) where
    mempty = Product 1
    Product x `mappend` Product y = Product (x * y)

getProduct $ Product 3 `mappend` Product 9  -- => 27

-- Sum (addition)
newtype Sum a = Sum { getSum :: a }
instance Num a => Monoid (Sum a) where
    mempty = Sum 0
    Sum x `mappend` Sum y = Sum (x + y)

getSum $ Sum 3 `mappend` Sum 9  -- => 12
```

**Any and All:**
```haskell
-- Any (Boolean OR)
newtype Any = Any { getAny :: Bool }
instance Monoid Any where
    mempty = Any False
    Any x `mappend` Any y = Any (x || y)

getAny $ Any True `mappend` Any False  -- => True

-- All (Boolean AND)
newtype All = All { getAll :: Bool }
instance Monoid All where
    mempty = All True
    All x `mappend` All y = All (x && y)

getAll $ All True `mappend` All True  -- => True
```

**Ordering Monoid:**
```haskell
instance Monoid Ordering where
    mempty = EQ
    LT `mappend` _ = LT
    EQ `mappend` y = y
    GT `mappend` _ = GT

-- Useful for comparing with multiple criteria
lengthCompare :: String -> String -> Ordering
lengthCompare x y = (length x `compare` length y) `mappend`
                    (x `compare` y)
```

**Maybe Monoid:**
```haskell
instance Monoid a => Monoid (Maybe a) where
    mempty = Nothing
    Nothing `mappend` m = m
    m `mappend` Nothing = m
    Just m1 `mappend` Just m2 = Just (m1 `mappend` m2)

Just (Sum 3) `mappend` Just (Sum 4)  -- => Just (Sum {getSum = 7})

-- First (first non-Nothing)
newtype First a = First { getFirst :: Maybe a }
instance Monoid (First a) where
    mempty = First Nothing
    First (Just x) `mappend` _ = First (Just x)
    First Nothing `mappend` x = x

getFirst $ First (Just 'a') `mappend` First Nothing  -- => Just 'a'
```

**Folding with Monoids:**
```haskell
-- foldMap
foldMap :: (Monoid m, Foldable t) => (a -> m) -> t a -> m

foldMap Sum [1,2,3,4]  -- => Sum {getSum = 10}
foldMap Product [1,2,3,4]  -- => Product {getProduct = 24}
foldMap Any [False,False,True]  -- => Any {getAny = True}
```

---

### Chapter 13: A Fistful of Monads

**The Monad Type Class:**
```haskell
class Monad m where
    return :: a -> m a
    (>>=) :: m a -> (a -> m b) -> m b
    (>>) :: m a -> m b -> m b
    x >> y = x >>= \_ -> y
    fail :: String -> m a
    fail msg = error msg

-- Maybe as a Monad
instance Monad Maybe where
    return x = Just x
    Nothing >>= f = Nothing
    Just x >>= f = f x
    fail _ = Nothing

-- Usage
Just 3 >>= (\x -> Just (x+1))  -- => Just 4
Nothing >>= (\x -> Just (x+1))  -- => Nothing
```

**Walking the Tightrope Example:**
```haskell
type Birds = Int
type Pole = (Birds, Birds)

landLeft :: Birds -> Pole -> Maybe Pole
landLeft n (left, right)
    | abs ((left + n) - right) < 4 = Just (left + n, right)
    | otherwise                     = Nothing

landRight :: Birds -> Pole -> Maybe Pole
landRight n (left, right)
    | abs (left - (right + n)) < 4 = Just (left, right + n)
    | otherwise                     = Nothing

-- Using >>= (bind)
return (0,0) >>= landLeft 1 >>= landRight 1 >>= landLeft 2
-- => Just (3,1)

return (0,0) >>= landLeft 1 >>= landRight 4 >>= landLeft (-1)
-- => Nothing

-- Using >> (then)
banana :: Pole -> Maybe Pole
banana _ = Nothing

return (0,0) >>= landLeft 1 >> banana >> landRight 1
-- => Nothing
```

**do Notation:**
```haskell
-- These are equivalent:
foo :: Maybe String
foo = Just 3 >>= (\x -> 
      Just "!" >>= (\y ->
      Just (show x ++ y)))

foo :: Maybe String
foo = do
    x <- Just 3
    y <- Just "!"
    Just (show x ++ y)

-- Pattern matching in do blocks
justH :: Maybe Char
justH = do
    (x:xs) <- Just "hello"
    return x
-- => Just 'h'

wopwop :: Maybe Char
wopwop = do
    (x:xs) <- Just ""
    return x
-- => Nothing
```

**List Monad:**
```haskell
instance Monad [] where
    return x = [x]
    xs >>= f = concat (map f xs)
    fail _ = []

-- Usage
[3,4,5] >>= \x -> [x,-x]  -- => [3,-3,4,-4,5,-5]
[1,2] >>= \n -> ['a','b'] >>= \ch -> return (n,ch)
-- => [(1,'a'),(1,'b'),(2,'a'),(2,'b')]

-- do notation with lists
listOfTuples :: [(Int,Char)]
listOfTuples = do
    n <- [1,2]
    ch <- ['a','b']
    return (n,ch)

-- List comprehensions are just syntactic sugar!
[x | x <- [1..50], '7' `elem` show x]
-- Same as:
[1..50] >>= (\x -> if '7' `elem` show x then return x else [])
```

**MonadPlus and guard:**
```haskell
class Monad m => MonadPlus m where
    mzero :: m a
    mplus :: m a -> m a -> m a

-- guard
guard :: MonadPlus m => Bool -> m ()
guard True = return ()
guard False = mzero

-- Usage
[1..50] >>= (\x -> guard ('7' `elem` show x) >> return x)

-- Knight's quest (chess problem)
type KnightPos = (Int, Int)

moveKnight :: KnightPos -> [KnightPos]
moveKnight (c,r) = do
    (c',r') <- [(c+2,r-1),(c+2,r+1),(c-2,r-1),(c-2,r+1)
               ,(c+1,r-2),(c+1,r+2),(c-1,r-2),(c-1,r+2)]
    guard (c' `elem` [1..8] && r' `elem` [1..8])
    return (c',r')

in3 :: KnightPos -> [KnightPos]
in3 start = return start >>= moveKnight >>= moveKnight >>= moveKnight

canReachIn3 :: KnightPos -> KnightPos -> Bool
canReachIn3 start end = end `elem` in3 start
```

**Monad Laws:**
```haskell
-- Left identity
return x >>= f  ≡  f x

-- Right identity
m >>= return  ≡  m

-- Associativity
(m >>= f) >>= g  ≡  m >>= (\x -> f x >>= g)
```

---

### Chapter 14: For a Few Monads More

**Writer Monad:**
```haskell
newtype Writer w a = Writer { runWriter :: (a, w) }

instance Monoid w => Monad (Writer w) where
    return x = Writer (x, mempty)
    (Writer (x,v)) >>= f = let (Writer (y, v')) = f x
                           in Writer (y, v `mappend` v')

-- tell function
tell :: Monoid w => w -> Writer w ()
tell w = Writer ((), w)

-- Usage
import Control.Monad.Writer

logNumber :: Int -> Writer [String] Int
logNumber x = Writer (x, ["Got number: " ++ show x])

multWithLog :: Writer [String] Int
multWithLog = do
    a <- logNumber 3
    b <- logNumber 5
    tell ["Gonna multiply these two"]
    return (a*b)

runWriter multWithLog  -- => (15,["Got number: 3","Got number: 5","Gonna multiply these two"])

-- Difference lists for efficient logging
newtype DiffList a = DiffList { getDiffList :: [a] -> [a] }

toDiffList :: [a] -> DiffList a
toDiffList xs = DiffList (xs++)

fromDiffList :: DiffList a -> [a]
fromDiffList (DiffList f) = f []
```

**Reader Monad:**
```haskell
instance Monad ((->) r) where
    return x = \_ -> x
    h >>= f = \w -> f (h w) w

-- Usage
addStuff :: Int -> Int
addStuff = do
    a <- (*2)
    b <- (+10)
    return (a+b)

addStuff 3  -- => 19
-- (*2) 3 = 6, (+10) 3 = 13, 6+13 = 19
```

**State Monad:**
```haskell
newtype State s a = State { runState :: s -> (a,s) }

instance Monad (State s) where
    return x = State $ \s -> (x,s)
    (State h) >>= f = State $ \s -> let (a, newState) = h s
                                        (State g) = f a
                                    in g newState

-- get and put
get = State $ \s -> (s,s)
put newState = State $ \s -> ((),newState)

-- Stack example
type Stack = [Int]

pop :: State Stack Int
pop = State $ \(x:xs) -> (x,xs)

push :: Int -> State Stack ()
push a = State $ \xs -> ((),a:xs)

stackManip :: State Stack Int
stackManip = do
    push 3
    a <- pop
    pop

runState stackManip [5,8,2,1]  -- => (5,[8,2,1])
```

**Error (Either) Monad:**
```haskell
instance (Error e) => Monad (Either e) where
    return x = Right x
    Right x >>= f = f x
    Left err >>= f = Left err
    fail msg = Left (strMsg msg)

-- Usage
Right 3 >>= \x -> return (x + 100)  -- => Right 103
Left "boom" >>= \x -> return (x + 100)  -- => Left "boom"
```

**Useful Monadic Functions:**
```haskell
-- liftM (like fmap)
liftM :: Monad m => (a -> b) -> m a -> m b
liftM f m = m >>= (\x -> return (f x))
-- or
liftM f m = do
    x <- m
    return (f x)

liftM (*3) (Just 8)  -- => Just 24

-- join (flattens nested monads)
join :: Monad m => m (m a) -> m a
join mm = do
    m <- mm
    m

join (Just (Just 9))  -- => Just 9
join [[1,2,3],[4,5,6]]  -- => [1,2,3,4,5,6]

-- filterM
filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]

-- Powerset example
powerset :: [a] -> [[a]]
powerset xs = filterM (\x -> [True, False]) xs

powerset [1,2,3]  -- => [[1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]]

-- foldM (monadic fold)
foldM :: Monad m => (a -> b -> m a) -> a -> [b] -> m a

binSmalls :: Int -> Int -> Maybe Int
binSmalls acc x
    | x > 9     = Nothing
    | otherwise = Just (acc + x)

foldM binSmalls 0 [2,8,3,1]  -- => Just 14
foldM binSmalls 0 [2,11,3,1]  -- => Nothing
```

**Safe RPN Calculator:**
```haskell
import Control.Monad

readMaybe :: Read a => String -> Maybe a
readMaybe st = case reads st of [(x,"")] -> Just x
                                 _ -> Nothing

foldingFunction :: [Double] -> String -> Maybe [Double]
foldingFunction (x:y:ys) "*" = return ((x * y):ys)
foldingFunction (x:y:ys) "+" = return ((x + y):ys)
foldingFunction (x:y:ys) "-" = return ((y - x):ys)
foldingFunction (x:y:ys) "/" = return ((y / x):ys)
foldingFunction xs numberString = liftM (:xs) (readMaybe numberString)

solveRPN :: String -> Maybe Double
solveRPN st = do
    [result] <- foldM foldingFunction [] (words st)
    return result
```

**Composing Monadic Functions:**
```haskell
-- Kleisli composition (>=>)
(>=>) :: Monad m => (a -> m b) -> (b -> m c) -> (a -> m c)
f >=> g = \x -> f x >>= g

-- Usage
let f = (+1) >=> (*100) >=> return . negate
f 10  -- => -1100
```

**Making Monads:**
```haskell
-- Example: Probability monad
import Data.Ratio

newtype Prob a = Prob { getProb :: [(a,Rational)] } deriving Show

instance Monad Prob where
    return x = Prob [(x,1%1)]
    m >>= f = Prob $ flatten $ map multAll $ getProb m
        where flatten = concat
              multAll (x,p) = map (\(y,p') -> (y,p*p')) $ getProb (f x)

-- Usage
thisSituation :: Prob (Prob Char)
thisSituation = Prob
    [( Prob [('a',1%2),('b',1%2)] , 1%4 )
    ,( Prob [('c',1%2),('d',1%2)] , 3%4 )
    ]

flatten :: Prob (Prob a) -> Prob a
flatten (Prob xs) = Prob $ concat $ map multAll xs
    where multAll (Prob innerxs,p) = map (\(x,r) -> (x,p*r)) innerxs
```

---

### Chapter 15: Zippers

**Zipper Data Structure:**
A zipper is a data structure that allows efficient navigation and modification of immutable data structures.

**Tree Zipper:**
```haskell
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

-- Breadcrumbs for navigation
data Crumb a = LeftCrumb a (Tree a) | RightCrumb a (Tree a) deriving (Show)
type Breadcrumbs a = [Crumb a]
type Zipper a = (Tree a, Breadcrumbs a)

-- Navigation functions
goLeft :: Zipper a -> Zipper a
goLeft (Node x l r, bs) = (l, LeftCrumb x r:bs)

goRight :: Zipper a -> Zipper a
goRight (Node x l r, bs) = (r, RightCrumb x l:bs)

goUp :: Zipper a -> Zipper a
goUp (t, LeftCrumb x r:bs) = (Node x t r, bs)
goUp (t, RightCrumb x l:bs) = (Node x l t, bs)

-- Modification
modify :: (a -> a) -> Zipper a -> Zipper a
modify f (Node x l r, bs) = (Node (f x) l r, bs)
modify f (Empty, bs) = (Empty, bs)

-- Attach new subtree
attach :: Tree a -> Zipper a -> Zipper a
attach t (_, bs) = (t, bs)

-- Go to top
topMost :: Zipper a -> Zipper a
topMost (t,[]) = (t,[])
topMost z = topMost (goUp z)

-- Example usage
freeTree = Node 'P'
    (Node 'O'
        (Node 'L'
            (Node 'N' Empty Empty)
            (Node 'T' Empty Empty))
        (Node 'Y'
            (Node 'S' Empty Empty)
            (Node 'A' Empty Empty)))
    (Node 'L'
        (Node 'W'
            (Node 'C' Empty Empty)
            (Node 'R' Empty Empty))
        (Node 'A'
            (Node 'A' Empty Empty)
            (Node 'C' Empty Empty)))

-- Change 'W' to 'P'
let newFocus = (freeTree,[]) -: goLeft -: goRight -: modify (\_ -> 'P')
```

**List Zipper:**
```haskell
type ListZipper a = ([a],[a])

goForward :: ListZipper a -> ListZipper a
goForward (x:xs, bs) = (xs, x:bs)

goBack :: ListZipper a -> ListZipper a
goBack (xs, b:bs) = (b:xs, bs)
```

**Filesystem Zipper:**
```haskell
data FSItem = File String Int | Folder String [FSItem] deriving (Show)

-- Filesystem zipper
data FSCrumb = FSCrumb String [FSItem] [FSItem] deriving (Show)
type FSZipper = (FSItem, [FSCrumb])

fsUp :: FSZipper -> FSZipper
fsUp (item, FSCrumb name ls rs:bs) = (Folder name (ls ++ [item] ++ rs), bs)

fsTo :: String -> FSZipper -> FSZipper
fsTo name (Folder folderName items, bs) =
    let (ls, item:rs) = break (nameIs name) items
    in (item, FSCrumb folderName ls rs:bs)

nameIs :: String -> FSItem -> Bool
nameIs name (Folder folderName _) = name == folderName
nameIs name (File fileName _) = name == fileName

-- Modification functions
fsRename :: String -> FSZipper -> FSZipper
fsRename newName (Folder name items, bs) = (Folder newName items, bs)
fsRename newName (File name size, bs) = (File newName size, bs)

fsNewFile :: FSItem -> FSZipper -> FSZipper
fsNewFile item (Folder folderName items, bs) =
    (Folder folderName (item:items), bs)
```

**Safe Navigation with Maybe:**
```haskell
goLeft :: Zipper a -> Maybe (Zipper a)
goLeft (Node x l r, bs) = Just (l, LeftCrumb x r:bs)
goLeft (Empty, _) = Nothing

goRight :: Zipper a -> Maybe (Zipper a)
goRight (Node x l r, bs) = Just (r, RightCrumb x l:bs)
goRight (Empty, _) = Nothing

-- Using >>= for chaining
coolTree -: return -: goRight >=> goLeft >=> goRight
```

---

## ADVANCED TOPICS & IN-DEPTH KNOWLEDGE

### Language Pragmas and Extensions

While "Learn You a Haskell" is a beginner's book and doesn't heavily cover language extensions, here are key extensions that advanced Haskellers use (mentioned briefly or implied in the book):

**Note:** The book doesn't extensively cover these, but they're important for production Haskell:

```haskell
-- Common extensions (not in book, but important to know)
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
```

### Type System Deep Dive

**Kinds** (covered in Chapter 7):
```haskell
-- Understanding kinds
:k Int           -- Int :: *
:k Maybe         -- Maybe :: * -> *
:k Either        -- Either :: * -> * -> *
:k Either String -- Either String :: * -> *

-- Type constructors must have kind * to be made instances of type classes
class Functor f where
    fmap :: (a -> b) -> f a -> f b
-- Here, f must have kind * -> *
```

### Lazy Evaluation Deep Dive

**Thunks and WHNF:**
```haskell
-- Lazy evaluation example
let xs = [1..1000000000]  -- Doesn't actually create the list
head xs  -- Only evaluates first element

-- Infinite data structures
ones = 1 : ones
numsFrom n = n : numsFrom (n+1)
squares = map (^2) (numsFrom 0)

-- Strictness
foldl vs foldl'  -- foldl' is strict, more memory efficient

-- Bang patterns (force evaluation)
data Point = Point !Int !Int  -- Fields are strict
```

### Performance Considerations

**Space Leaks:**
```haskell
-- Bad: space leak with foldl
sum' = foldl (+) 0

-- Good: strict fold
import Data.List (foldl')
sum' = foldl' (+) 0

-- Strictness in data types
data Vec = Vec !Double !Double !Double
```

### Functor, Applicative, Monad Hierarchy

```haskell
-- The hierarchy (as it should be understood):
class Functor f where
    fmap :: (a -> b) -> f a -> f b

class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

class Applicative m => Monad m where
    return :: a -> m a  -- Should be same as pure
    (>>=) :: m a -> (a -> m b) -> m b

-- Every Monad is an Applicative
-- Every Applicative is a Functor
```

### Type Class Deriving Strategies

```haskell
-- Simple deriving
data Color = Red | Green | Blue deriving (Show, Eq, Ord)

-- Newtype deriving
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
newtype Age = Age Int deriving (Show, Eq, Ord, Num)

-- DeriveFunctor
{-# LANGUAGE DeriveFunctor #-}
data Tree a = Leaf a | Node (Tree a) (Tree a) deriving Functor
```

### Best Practices from the Book

1. **Use types to guide development**
2. **Start with simple, obviously correct functions**
3. **Compose functions to build complexity**
4. **Use pattern matching liberally**
5. **Leverage laziness for infinite structures**
6. **Prefer pure functions over I/O**
7. **Use type classes for polymorphism**
8. **Think recursively**
9. **Use higher-order functions (map, filter, fold) instead of explicit recursion**
10. **Make illegal states unrepresentable with types**

### Common Patterns

**Option/Maybe Pattern:**
```haskell
-- Instead of null checks
lookup :: Eq a => a -> [(a,b)] -> Maybe b
safehead :: [a] -> Maybe a
safediv :: Int -> Int -> Maybe Int
```

**Either for Error Handling:**
```haskell
data ParseError = ParseError String
parseValue :: String -> Either ParseError Value
```

**Reader Pattern for Configuration:**
```haskell
type Config = String
type App a = Reader Config a

getConfig :: App Config
getConfig = ask
```

**State Pattern for Stateful Computations:**
```haskell
type GameState = State World ()

movePlayer :: Direction -> GameState
updateScore :: Int -> GameState
```

---

## KEY TAKEAWAYS

1. **Haskell is purely functional** - functions have no side effects
2. **Lazy evaluation** - expressions evaluated only when needed
3. **Strong static typing** with type inference
4. **Type classes** provide ad-hoc polymorphism
5. **Algebraic data types** for modeling data
6. **Pattern matching** for deconstructing data
7. **Higher-order functions** are first-class citizens
8. **Monads** for sequencing computations with context
9. **Functors/Applicatives** for mapping over contexts
10. **Composition** is key to building complex programs

---

## FURTHER TOPICS NOT COVERED

The book is comprehensive for beginners but doesn't cover:
- Advanced type system features (GADTs, Type Families, Dependent Types)
- Template Haskell (metaprogramming)
- Lens library
- Concurrent and parallel programming
- Streaming libraries (Conduit, Pipes)
- Web frameworks
- Database access
- Foreign Function Interface (FFI)
- Profiling and optimization
- Package management with Cabal/Stack
- Advanced Monad Transformers

This summary covers all major topics from the 400+ page book, with code examples and explanations for each concept!
