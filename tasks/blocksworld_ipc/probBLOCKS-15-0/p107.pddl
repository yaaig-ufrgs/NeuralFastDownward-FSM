(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear d)
		(clear e)
		(clear n)
		(handempty)
		(on b k)
		(on c g)
		(on d l)
		(on f h)
		(on h i)
		(on i m)
		(on j o)
		(on k f)
		(on l c)
		(on n j)
		(on o b)
		(ontable a)
		(ontable e)
		(ontable g)
		(ontable m)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
