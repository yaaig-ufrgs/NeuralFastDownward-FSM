(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear b)
	(clear f)
	(handempty)
	(on a d)
	(on b c)
	(on c a)
	(on e g)
	(on f e)
	(ontable d)
	(ontable g)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
