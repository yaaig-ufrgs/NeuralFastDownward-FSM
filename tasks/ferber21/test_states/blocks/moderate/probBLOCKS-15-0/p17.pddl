(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear d)
		(clear f)
		(clear h)
		(clear j)
		(clear n)
		(handempty)
		(on a l)
		(on b e)
		(on c b)
		(on d o)
		(on e m)
		(on f i)
		(on g k)
		(on h c)
		(on m a)
		(on n g)
		(ontable i)
		(ontable j)
		(ontable k)
		(ontable l)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
