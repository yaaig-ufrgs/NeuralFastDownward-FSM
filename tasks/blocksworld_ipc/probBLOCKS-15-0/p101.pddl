(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear b)
		(clear f)
		(clear n)
		(handempty)
		(on a e)
		(on c g)
		(on d j)
		(on e h)
		(on f i)
		(on h d)
		(on i o)
		(on j m)
		(on k l)
		(on l c)
		(on m k)
		(ontable b)
		(ontable g)
		(ontable n)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
