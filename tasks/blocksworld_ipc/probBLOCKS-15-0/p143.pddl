(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear b)
		(clear n)
		(clear o)
		(handempty)
		(on a m)
		(on b f)
		(on c g)
		(on d l)
		(on e j)
		(on h k)
		(on i h)
		(on j d)
		(on k e)
		(on l c)
		(on m i)
		(on n a)
		(ontable f)
		(ontable g)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
