(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear b)
		(clear f)
		(clear j)
		(clear n)
		(handempty)
		(on a c)
		(on b d)
		(on e l)
		(on f o)
		(on g m)
		(on j h)
		(on k i)
		(on m k)
		(on n g)
		(on o e)
		(ontable c)
		(ontable d)
		(ontable h)
		(ontable i)
		(ontable l)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
