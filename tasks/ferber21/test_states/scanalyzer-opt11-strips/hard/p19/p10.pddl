(define (problem scanalyzer3d-42)
  (:domain scanalyzer3d)
  (:objects
    car-in-1 - car
    car-in-2 - car
    car-in-3 - car
    car-in-4 - car
    car-in-5 - car
    car-in-6 - car
    car-in-7 - car
    car-in-8 - car
    car-in-9 - car
    car-out-1 - car
    car-out-2 - car
    car-out-3 - car
    car-out-4 - car
    car-out-5 - car
    car-out-6 - car
    car-out-7 - car
    car-out-8 - car
    car-out-9 - car
    seg-in-1 - segment
    seg-in-2 - segment
    seg-in-3 - segment
    seg-in-4 - segment
    seg-in-5 - segment
    seg-in-6 - segment
    seg-in-7 - segment
    seg-in-8 - segment
    seg-in-9 - segment
    seg-out-1 - segment
    seg-out-2 - segment
    seg-out-3 - segment
    seg-out-4 - segment
    seg-out-5 - segment
    seg-out-6 - segment
    seg-out-7 - segment
    seg-out-8 - segment
    seg-out-9 - segment
  )
  (:init
	(analyzed car-in-1)
	(analyzed car-in-2)
	(analyzed car-in-3)
	(analyzed car-in-4)
	(analyzed car-in-6)
	(analyzed car-in-7)
	(analyzed car-in-8)
	(analyzed car-out-1)
	(analyzed car-out-2)
	(analyzed car-out-3)
	(analyzed car-out-4)
	(analyzed car-out-5)
	(analyzed car-out-6)
	(analyzed car-out-7)
	(analyzed car-out-8)
	(on car-in-1 seg-in-2)
	(on car-in-2 seg-out-1)
	(on car-in-3 seg-in-6)
	(on car-in-4 seg-out-9)
	(on car-in-5 seg-in-1)
	(on car-in-6 seg-in-4)
	(on car-in-7 seg-in-3)
	(on car-in-8 seg-in-9)
	(on car-in-9 seg-out-7)
	(on car-out-1 seg-in-5)
	(on car-out-2 seg-out-5)
	(on car-out-3 seg-out-8)
	(on car-out-4 seg-in-8)
	(on car-out-5 seg-out-4)
	(on car-out-6 seg-in-7)
	(on car-out-7 seg-out-3)
	(on car-out-8 seg-out-6)
	(on car-out-9 seg-out-2)
	(= (total-cost) 0)
	(cycle-2 seg-in-1 seg-out-1)
	(cycle-2 seg-in-1 seg-out-2)
	(cycle-2 seg-in-1 seg-out-3)
	(cycle-2 seg-in-1 seg-out-4)
	(cycle-2 seg-in-1 seg-out-5)
	(cycle-2 seg-in-1 seg-out-6)
	(cycle-2 seg-in-1 seg-out-7)
	(cycle-2 seg-in-1 seg-out-8)
	(cycle-2 seg-in-1 seg-out-9)
	(cycle-2 seg-in-2 seg-out-1)
	(cycle-2 seg-in-2 seg-out-2)
	(cycle-2 seg-in-2 seg-out-3)
	(cycle-2 seg-in-2 seg-out-4)
	(cycle-2 seg-in-2 seg-out-5)
	(cycle-2 seg-in-2 seg-out-6)
	(cycle-2 seg-in-2 seg-out-7)
	(cycle-2 seg-in-2 seg-out-8)
	(cycle-2 seg-in-2 seg-out-9)
	(cycle-2 seg-in-3 seg-out-1)
	(cycle-2 seg-in-3 seg-out-2)
	(cycle-2 seg-in-3 seg-out-3)
	(cycle-2 seg-in-3 seg-out-4)
	(cycle-2 seg-in-3 seg-out-5)
	(cycle-2 seg-in-3 seg-out-6)
	(cycle-2 seg-in-3 seg-out-7)
	(cycle-2 seg-in-3 seg-out-8)
	(cycle-2 seg-in-3 seg-out-9)
	(cycle-2 seg-in-4 seg-out-1)
	(cycle-2 seg-in-4 seg-out-2)
	(cycle-2 seg-in-4 seg-out-3)
	(cycle-2 seg-in-4 seg-out-4)
	(cycle-2 seg-in-4 seg-out-5)
	(cycle-2 seg-in-4 seg-out-6)
	(cycle-2 seg-in-4 seg-out-7)
	(cycle-2 seg-in-4 seg-out-8)
	(cycle-2 seg-in-4 seg-out-9)
	(cycle-2 seg-in-5 seg-out-1)
	(cycle-2 seg-in-5 seg-out-2)
	(cycle-2 seg-in-5 seg-out-3)
	(cycle-2 seg-in-5 seg-out-4)
	(cycle-2 seg-in-5 seg-out-5)
	(cycle-2 seg-in-5 seg-out-6)
	(cycle-2 seg-in-5 seg-out-7)
	(cycle-2 seg-in-5 seg-out-8)
	(cycle-2 seg-in-5 seg-out-9)
	(cycle-2 seg-in-6 seg-out-1)
	(cycle-2 seg-in-6 seg-out-2)
	(cycle-2 seg-in-6 seg-out-3)
	(cycle-2 seg-in-6 seg-out-4)
	(cycle-2 seg-in-6 seg-out-5)
	(cycle-2 seg-in-6 seg-out-6)
	(cycle-2 seg-in-6 seg-out-7)
	(cycle-2 seg-in-6 seg-out-8)
	(cycle-2 seg-in-6 seg-out-9)
	(cycle-2 seg-in-7 seg-out-1)
	(cycle-2 seg-in-7 seg-out-2)
	(cycle-2 seg-in-7 seg-out-3)
	(cycle-2 seg-in-7 seg-out-4)
	(cycle-2 seg-in-7 seg-out-5)
	(cycle-2 seg-in-7 seg-out-6)
	(cycle-2 seg-in-7 seg-out-7)
	(cycle-2 seg-in-7 seg-out-8)
	(cycle-2 seg-in-7 seg-out-9)
	(cycle-2 seg-in-8 seg-out-1)
	(cycle-2 seg-in-8 seg-out-2)
	(cycle-2 seg-in-8 seg-out-3)
	(cycle-2 seg-in-8 seg-out-4)
	(cycle-2 seg-in-8 seg-out-5)
	(cycle-2 seg-in-8 seg-out-6)
	(cycle-2 seg-in-8 seg-out-7)
	(cycle-2 seg-in-8 seg-out-8)
	(cycle-2 seg-in-8 seg-out-9)
	(cycle-2 seg-in-9 seg-out-1)
	(cycle-2 seg-in-9 seg-out-2)
	(cycle-2 seg-in-9 seg-out-3)
	(cycle-2 seg-in-9 seg-out-4)
	(cycle-2 seg-in-9 seg-out-5)
	(cycle-2 seg-in-9 seg-out-6)
	(cycle-2 seg-in-9 seg-out-7)
	(cycle-2 seg-in-9 seg-out-8)
	(cycle-2 seg-in-9 seg-out-9)
	(cycle-2-with-analysis seg-in-1 seg-out-1)
	(cycle-2-with-analysis seg-in-2 seg-out-1)
	(cycle-2-with-analysis seg-in-3 seg-out-1)
	(cycle-2-with-analysis seg-in-4 seg-out-1)
	(cycle-2-with-analysis seg-in-5 seg-out-1)
	(cycle-2-with-analysis seg-in-6 seg-out-1)
	(cycle-2-with-analysis seg-in-7 seg-out-1)
	(cycle-2-with-analysis seg-in-8 seg-out-1)
	(cycle-2-with-analysis seg-in-9 seg-out-1)
)
  (:goal (and
    (analyzed car-in-1)
    (analyzed car-in-2)
    (analyzed car-in-3)
    (analyzed car-in-4)
    (analyzed car-in-5)
    (analyzed car-in-6)
    (analyzed car-in-7)
    (analyzed car-in-8)
    (analyzed car-in-9)
    (analyzed car-out-1)
    (analyzed car-out-2)
    (analyzed car-out-3)
    (analyzed car-out-4)
    (analyzed car-out-5)
    (analyzed car-out-6)
    (analyzed car-out-7)
    (analyzed car-out-8)
    (analyzed car-out-9)
    (on car-in-1 seg-in-1)
    (on car-in-2 seg-in-2)
    (on car-in-3 seg-in-3)
    (on car-in-4 seg-in-4)
    (on car-in-5 seg-in-5)
    (on car-in-6 seg-in-6)
    (on car-in-7 seg-in-7)
    (on car-in-8 seg-in-8)
    (on car-in-9 seg-in-9)
    (on car-out-1 seg-out-1)
    (on car-out-2 seg-out-2)
    (on car-out-3 seg-out-3)
    (on car-out-4 seg-out-4)
    (on car-out-5 seg-out-5)
    (on car-out-6 seg-out-6)
    (on car-out-7 seg-out-7)
    (on car-out-8 seg-out-8)
    (on car-out-9 seg-out-9)
  ))
  
 (:metric minimize (total-cost))
)

