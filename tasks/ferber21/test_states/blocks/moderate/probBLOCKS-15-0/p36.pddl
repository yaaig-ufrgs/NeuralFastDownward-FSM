(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear c)
		(clear k)
		(handempty)
		(on a i)
		(on b a)
		(on c d)
		(on d g)
		(on f b)
		(on g l)
		(on h f)
		(on i m)
		(on j e)
		(on k j)
		(on l o)
		(on m n)
		(on o h)
		(ontable e)
		(ontable n)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
