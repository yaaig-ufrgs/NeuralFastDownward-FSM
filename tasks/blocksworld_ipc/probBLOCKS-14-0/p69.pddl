(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear h)
		(handempty)
		(on a k)
		(on b a)
		(on c f)
		(on d i)
		(on e j)
		(on f g)
		(on g d)
		(on h e)
		(on i n)
		(on j m)
		(on k c)
		(on l b)
		(on m l)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
