(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear c)
		(clear i)
		(clear l)
		(handempty)
		(on a o)
		(on b a)
		(on c f)
		(on d j)
		(on e b)
		(on f d)
		(on g n)
		(on i k)
		(on j m)
		(on k h)
		(on l e)
		(on m g)
		(ontable h)
		(ontable n)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
