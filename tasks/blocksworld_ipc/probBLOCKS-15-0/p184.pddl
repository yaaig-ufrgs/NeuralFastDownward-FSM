(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear d)
		(clear e)
		(clear o)
		(handempty)
		(on a c)
		(on b f)
		(on c g)
		(on d a)
		(on f k)
		(on g l)
		(on h i)
		(on i b)
		(on j m)
		(on l n)
		(on n h)
		(on o j)
		(ontable e)
		(ontable k)
		(ontable m)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
