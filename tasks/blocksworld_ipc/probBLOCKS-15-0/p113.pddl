(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear b)
		(clear i)
		(clear m)
		(clear o)
		(handempty)
		(on a e)
		(on c g)
		(on d l)
		(on e j)
		(on h k)
		(on i n)
		(on j d)
		(on l c)
		(on m h)
		(on n f)
		(on o a)
		(ontable b)
		(ontable f)
		(ontable g)
		(ontable k)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
