(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear b)
		(clear g)
		(clear i)
		(clear j)
		(clear l)
		(clear m)
		(handempty)
		(on b n)
		(on c e)
		(on f k)
		(on g o)
		(on h a)
		(on j h)
		(on k c)
		(on m d)
		(on n f)
		(ontable a)
		(ontable d)
		(ontable e)
		(ontable i)
		(ontable l)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
