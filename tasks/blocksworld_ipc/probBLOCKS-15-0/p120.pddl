(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear i)
		(clear j)
		(clear k)
		(handempty)
		(on a o)
		(on b d)
		(on d h)
		(on e f)
		(on f m)
		(on h c)
		(on i e)
		(on j l)
		(on k n)
		(on l b)
		(on m a)
		(on n g)
		(ontable c)
		(ontable g)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
