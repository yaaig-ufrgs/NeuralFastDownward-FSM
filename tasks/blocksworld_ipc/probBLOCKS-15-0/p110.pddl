(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear h)
		(clear m)
		(clear n)
		(handempty)
		(on a b)
		(on b f)
		(on c g)
		(on d l)
		(on e j)
		(on f d)
		(on i o)
		(on j a)
		(on k i)
		(on l c)
		(on m k)
		(on n e)
		(ontable g)
		(ontable h)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
